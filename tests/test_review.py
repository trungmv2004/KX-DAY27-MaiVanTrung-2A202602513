"""Offline behavior checks: no API calls, credentials, or GitHub writes."""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.types import Command
from rich.console import Console

import app as ui
from audit.replay import list_threads, replay
from common.db import replay_events
from common.schemas import PRAnalysis, ReviewComment
from exercises import exercise_1_confidence as ex1
from exercises import exercise_2_hitl as ex2
from exercises import exercise_3_escalation as ex3
from exercises import exercise_4_audit as ex4
from exercises.review_helpers import post_review_with_url


URL = "https://github.com/example/demo/pull/1"
QUESTION = "In auth.py:2, why use MD5 for password hashing?"
PR = SimpleNamespace(
    title="Update auth", diff="diff --git a/auth.py b/auth.py\n+hash = md5(password)",
    files_changed=["auth.py"], head_sha="abcdef1234567",
)


def analysis(confidence, questions=False):
    return PRAnalysis(
        summary="Review of auth changes", confidence=confidence,
        confidence_reasoning="Missing context about password hashing",
        risk_factors=["Password hashing requires review"],
        comments=[ReviewComment(file="auth.py", line=2, severity="blocker", body="Use Argon2.")],
        escalation_questions=[QUESTION] if questions else [],
    )


class FakeLLM:
    def __init__(self, *results):
        self.results = iter(results)
        self.calls = []

    def with_structured_output(self, schema):
        assert schema is PRAnalysis
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return next(self.results)

    async def ainvoke(self, messages):
        return self.invoke(messages)


class EarlyExercises(unittest.TestCase):
    def setUp(self):
        for module in (ex1, ex2, ex3):
            patcher = patch.object(module, "console", Console(file=io.StringIO()))
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_routing_boundaries(self):
        for module in (ex1, ex2, ex3):
            for confidence, expected in (
                (0.0, "escalate"), (0.5799, "escalate"),
                (0.58, "human_approval"), (0.72, "human_approval"),
                (0.7299, "human_approval"), (0.73, "auto_approve"), (1.0, "auto_approve"),
            ):
                with self.subTest(module=module.__name__, confidence=confidence):
                    self.assertEqual(module.node_route({"analysis": analysis(confidence)})["decision"], expected)

    def test_exercise_one_graph(self):
        for confidence, action in ((0.9, "auto_approved"), (0.65, "pending_human_approval"), (0.4, "pending_escalation")):
            with patch.object(ex1, "fetch_pr", return_value=PR), patch.object(ex1, "get_llm", return_value=FakeLLM(analysis(confidence))):
                self.assertEqual(ex1.build_graph().invoke({"pr_url": URL})["final_action"], action)

    def test_approval_reject_and_edit(self):
        for choice in ("approve", "reject", "edit"):
            with patch.object(ex2, "fetch_pr", return_value=PR), patch.object(ex2, "get_llm", return_value=FakeLLM(analysis(0.65))), patch.object(ex2, "post_review_comment") as post:
                graph = ex2.build_graph()
                cfg = {"configurable": {"thread_id": choice}}
                paused = graph.invoke({"pr_url": URL}, cfg)
                self.assertEqual(paused["__interrupt__"][0].value["kind"], "approval_request")
                post.assert_not_called()
                result = graph.invoke(Command(resume={"choice": choice, "feedback": "Edited review"}), cfg)
                self.assertEqual(post.call_count, int(choice != "reject"))
                self.assertEqual(result["final_action"], "rejected" if choice == "reject" else "committed")
                if choice == "edit":
                    self.assertEqual(post.call_args.args[1], "Edited review")

    def test_escalation_requires_final_confirmation(self):
        with patch.object(ex3, "fetch_pr", return_value=PR), patch.object(ex3, "get_llm", return_value=FakeLLM(analysis(0.4, True), analysis(0.8))), patch.object(ex3, "post_review_comment") as post:
            graph = ex3.build_graph()
            cfg = {"configurable": {"thread_id": "escalation"}}
            graph.invoke({"pr_url": URL}, cfg)
            refined = graph.invoke(Command(resume={QUESTION: "We will replace MD5."}), cfg)
            self.assertEqual(refined["__interrupt__"][0].value["kind"], "approval_request")
            post.assert_not_called()
            result = graph.invoke(Command(resume={"choice": "reject"}), cfg)
            self.assertEqual(result["final_action"], "rejected")
            post.assert_not_called()


class DurableReview(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, {"HITL_DB_PATH": str(Path(self.temp.name) / "audit.db"), "GITHUB_USER": "reviewer"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.fetch = patch.object(ex4, "fetch_pr", return_value=PR).start()
        self.post = patch.object(ex4, "post_review_comment", return_value=URL + "#issuecomment-123").start()
        self.llm = FakeLLM(analysis(0.65))
        patch.object(ex4, "get_llm", side_effect=lambda **kwargs: self.llm).start()
        patch.object(ex4, "console", Console(file=io.StringIO())).start()
        self.addCleanup(patch.stopall)

    async def test_restart_approval_and_no_duplicate_audit_or_post(self):
        first = await ui.run_graph(URL, "approval")
        self.assertIn("__interrupt__", first)
        self.post.assert_not_called()
        # Each invocation closes the saver and creates another connection.
        restored = await ui.load_session("approval")
        self.assertEqual(restored["payload"]["kind"], "approval_request")
        await ui.run_graph(URL, "approval")
        final = await ui.run_graph(URL, "approval", {"choice": "approve"})
        self.assertEqual(final["final_action"], "committed")
        await ui.run_graph(URL, "approval")
        self.assertEqual(self.fetch.call_count, 1)
        self.post.assert_called_once()
        events = await replay_events("approval")
        self.assertEqual([e["action"] for e in events], ["fetch_pr", "analyze", "route", "human_approval", "human_approval", "commit"])
        self.assertEqual(events[-2]["reviewer_id"], "reviewer")
        self.assertEqual(events[-2]["decision"], "approve")

    async def test_edit_posts_exact_body(self):
        await ui.run_graph(URL, "edit")
        result = await ui.run_graph(URL, "edit", {"choice": "edit", "feedback": "Reviewer corrected this review."})
        self.assertEqual(result["posted_comment_body"], "Reviewer corrected this review.")
        self.assertEqual(self.post.call_args.args[1], result["posted_comment_body"])

    async def test_reject_never_posts(self):
        await ui.run_graph(URL, "reject")
        result = await ui.run_graph(URL, "reject", {"choice": "reject", "feedback": "Insufficient evidence"})
        self.assertEqual(result["final_action"], "rejected")
        self.post.assert_not_called()

    async def test_escalation_refines_and_waits_for_confirmation(self):
        self.llm = FakeLLM(analysis(0.4, True), analysis(0.85))
        paused = await ui.run_graph(URL, "escalation")
        self.assertEqual(paused["__interrupt__"][0].value["questions"], [QUESTION])
        refined = await ui.run_graph(URL, "escalation", {QUESTION: "MD5 is legacy; replace it."})
        self.assertTrue(refined["__interrupt__"][0].value["refined"])
        self.post.assert_not_called()
        self.assertIn("MD5 is legacy", self.llm.calls[1][1]["content"])
        result = await ui.run_graph(URL, "escalation", {"choice": "approve"})
        self.assertEqual(result["final_action"], "committed")
        events = await replay_events("escalation")
        self.assertEqual([e["decision"] for e in events if e["action"] == "escalate"], ["pending", "escalate"])
        self.assertIn("MD5 is legacy", next(e["reason"] for e in events if e["action"] == "escalate" and e["decision"] == "escalate"))
        sessions = await list_threads(display=False)
        self.assertEqual(sessions[0]["worst_risk"], "high")
        await replay("escalation")

    async def test_auto_approve_and_failure_status(self):
        self.llm = FakeLLM(analysis(0.9), analysis(0.9))
        result = await ui.run_graph(URL, "auto")
        self.assertEqual(result["final_action"], "auto_approved")
        self.post.side_effect = RuntimeError("GitHub unavailable")
        failed = await ui.run_graph(URL, "failed")
        self.assertEqual(failed["final_action"], "commit_failed")
        self.assertIsNone(failed["posted_comment_body"])

    async def test_validation_keeps_pause_and_prevents_writes(self):
        await ui.run_graph(URL, "validation")
        for answer in ({"choice": "other"}, {"choice": "edit", "feedback": " "}):
            with self.assertRaises(ValueError):
                await ui.run_graph(URL, "validation", answer)
        self.post.assert_not_called()
        with self.assertRaises(ValueError):
            await ui.run_graph(URL.replace("pull/1", "pull/2"), "validation")

    async def test_fallback_questions_and_empty_answers(self):
        self.llm = FakeLLM(analysis(0.4))
        paused = await ui.run_graph(URL, "fallback")
        question = paused["__interrupt__"][0].value["questions"][0]
        self.assertIn("auth.py:2", question)
        with self.assertRaises(ValueError):
            await ui.run_graph(URL, "fallback", {question: " "})
        self.post.assert_not_called()


class StreamlitReview(unittest.TestCase):
    def test_github_comment_link(self):
        import httpx

        def handle(request):
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/repos/example/demo/issues/1/comments")
            self.assertEqual(json.loads(request.content), {"body": "Reviewed"})
            return httpx.Response(201, json={"html_url": URL + "#issuecomment-123"})

        client = httpx.Client(transport=httpx.MockTransport(handle))
        with patch.dict(os.environ, {"GITHUB_TOKEN": "offline-test-token"}), patch("httpx.Client", return_value=client):
            self.assertEqual(post_review_with_url(URL, "Reviewed"), URL + "#issuecomment-123")

    def test_ui_approval_and_browser_reload(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"HITL_DB_PATH": str(Path(directory) / "ui.db")}), patch.object(ex4, "fetch_pr", return_value=PR), patch.object(ex4, "get_llm", return_value=FakeLLM(analysis(0.65))), patch.object(ex4, "post_review_comment", return_value=URL + "#issuecomment-123") as post:
            script = str(Path(ui.__file__).resolve())
            page = AppTest.from_file(script, default_timeout=20).run()
            page.text_input[0].set_value(URL)
            next(b for b in page.button if b.label == "Run review").click().run()
            self.assertFalse(page.exception)
            thread_id = page.session_state["thread_id"]
            self.assertTrue(any("Approval requested" in h.value for h in page.subheader))
            # Simulate an entirely new browser session with the persisted URL.
            restored = AppTest.from_file(script, default_timeout=20)
            restored.query_params["thread"] = thread_id
            restored.run()
            self.assertFalse(restored.exception)
            next(b for b in restored.button if b.label == "Approve").click().run()
            self.assertFalse(restored.exception)
            self.assertTrue(restored.success)
            post.assert_called_once()
            restored.run()
            post.assert_called_once()

    def test_ui_escalation_answers_and_edited_confirmation(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"HITL_DB_PATH": str(Path(directory) / "ui.db")}), patch.object(ex4, "fetch_pr", return_value=PR), patch.object(ex4, "get_llm", return_value=FakeLLM(analysis(0.4, True), analysis(0.8))), patch.object(ex4, "post_review_comment", return_value=URL + "#issuecomment-123") as post:
            page = AppTest.from_file(str(Path(ui.__file__).resolve()), default_timeout=20).run()
            page.text_input[0].set_value(URL)
            next(b for b in page.button if b.label == "Run review").click().run()
            self.assertFalse(page.exception)
            next(b for b in page.button if b.label == "Submit answers").click().run()
            self.assertTrue(any("answer every question" in e.value for e in page.error))
            page.text_area[0].set_value("Replace MD5 with Argon2.")
            next(b for b in page.button if b.label == "Submit answers").click().run()
            self.assertFalse(page.exception)
            self.assertTrue(any("Confirm refined review" in h.value for h in page.subheader))
            post.assert_not_called()
            page.text_area[0].set_value("Final corrected review")
            next(b for b in page.button if b.label == "Edit").click().run()
            self.assertFalse(page.exception)
            self.assertTrue(page.success)
            self.assertEqual(post.call_args.args[1], "Final corrected review")

    def test_ui_auto_post_failure_is_error(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"HITL_DB_PATH": str(Path(directory) / "ui.db")}), patch.object(ex4, "fetch_pr", return_value=PR), patch.object(ex4, "get_llm", return_value=FakeLLM(analysis(0.9))), patch.object(ex4, "post_review_comment", side_effect=RuntimeError("GitHub unavailable")):
            page = AppTest.from_file(str(Path(ui.__file__).resolve()), default_timeout=20).run()
            page.text_input[0].set_value(URL)
            next(b for b in page.button if b.label == "Run review").click().run()
            self.assertFalse(page.exception)
            self.assertFalse(page.success)
            self.assertTrue(any("GitHub unavailable" in e.value for e in page.error))


if __name__ == "__main__":
    unittest.main()
