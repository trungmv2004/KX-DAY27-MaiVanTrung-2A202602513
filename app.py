"""Exercise 5: run with `uv run streamlit run app.py`."""

from __future__ import annotations

import asyncio
import re
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from audit.replay import list_threads
from common.db import db_path
from exercises.exercise_4_audit import build_graph, invoke_session


async def recent_sessions() -> list[dict]:
    return await list_threads(display=False)


async def load_session(thread_id: str) -> dict:
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        snapshot = await build_graph(cp).aget_state({"configurable": {"thread_id": thread_id}})
    interrupts = [i for task in snapshot.tasks for i in task.interrupts]
    return {
        "values": snapshot.values,
        "payload": interrupts[0].value if interrupts else None,
        "unfinished": bool(snapshot.next),
    }


async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        return await invoke_session(build_graph(cp), pr_url, thread_id, resume_value)


def remember_result(result: dict) -> None:
    interrupts = result.get("__interrupt__", ())
    st.session_state.interrupt_payload = interrupts[0].value if interrupts else None
    st.session_state.final = None if interrupts else result
    st.session_state.unfinished = False


def restore_session(thread_id: str) -> None:
    session = asyncio.run(load_session(thread_id))
    if not session["values"]:
        raise ValueError("Session not found. Choose a recent session or start a review.")
    st.session_state.thread_id = thread_id
    st.session_state.pr_url = session["values"]["pr_url"]
    st.session_state.interrupt_payload = session["payload"]
    st.session_state.unfinished = session["unfinished"]
    st.session_state.final = None if session["unfinished"] else session["values"]
    st.query_params["thread"] = thread_id


def render_approval_card(payload: dict) -> dict | None:
    label = "Confirm refined review" if payload.get("refined") else "Approval requested"
    st.subheader(f"{label} — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])
    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")
    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")
    key = f"{st.session_state.thread_id}_{bool(payload.get('refined'))}"
    feedback = st.text_input("Reviewer note (optional)", key=f"note_{key}")
    edited = st.text_area(
        "Edited review body (used by Edit)", value=payload.get("review_body", ""),
        height=200, key=f"edit_{key}",
    )
    approve, reject, edit = st.columns(3)
    if approve.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if reject.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if edit.button("Edit"):
        if edited.strip():
            return {"choice": "edit", "feedback": edited.strip()}
        st.error("Enter the edited review before submitting.")
    return None


def render_escalation_card(payload: dict) -> dict | None:
    st.subheader(f"Strong escalation — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])
    for risk in payload.get("risk_factors", []):
        st.error(risk)
    with st.form(f"escalation_{st.session_state.thread_id}"):
        answers = {
            q: st.text_area(q, key=f"q_{st.session_state.thread_id}_{i}")
            for i, q in enumerate(payload["questions"])
        }
        if st.form_submit_button("Submit answers"):
            if all(answer.strip() for answer in answers.values()):
                return {q: answer.strip() for q, answer in answers.items()}
            st.error("Please answer every question.")
    return None


def advance(resume_value=None) -> None:
    try:
        with st.spinner("Review in progress..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value,
            ))
        remember_result(result)
        st.rerun()
    except Exception as exc:
        st.error(f"Review could not continue: {exc}")
        # The answer may already be checkpointed before a downstream failure.
        restore_session(st.session_state.thread_id)


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="HITL PR Review", layout="wide")
    st.title("HITL PR Review Agent")
    for key, default in {
        "thread_id": None, "pr_url": "", "interrupt_payload": None,
        "final": None, "unfinished": False,
    }.items():
        st.session_state.setdefault(key, default)

    saved_thread = st.query_params.get("thread")
    if saved_thread and saved_thread != st.session_state.thread_id:
        try:
            restore_session(saved_thread)
        except Exception as exc:
            st.error(str(exc))

    with st.sidebar:
        st.header("Recent sessions")
        try:
            sessions = asyncio.run(recent_sessions())
            if sessions:
                by_id = {s["thread_id"]: s for s in sessions}
                selected = st.selectbox(
                    "Session", list(by_id), index=None,
                    format_func=lambda tid: (
                        f"{tid[:8]} · PR #{by_id[tid]['pr_url'].rsplit('/', 1)[-1]}"
                        f" · {by_id[tid]['worst_risk']}"
                    ),
                )
                if st.button("Open session", disabled=selected is None):
                    restore_session(selected)
                    st.rerun()
            else:
                st.caption("No sessions yet — run a review.")
        except Exception as exc:
            st.error(f"Could not load sessions: {exc}")

    with st.form("start"):
        pr_url = st.text_input(
            "PR URL", value=st.session_state.pr_url,
            placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
        )
        submitted = st.form_submit_button("Run review")
    if submitted:
        pr_url = pr_url.strip()
        if re.fullmatch(r"https://github\.com/[^/\s]+/[^/\s]+/pull/[1-9]\d*/?", pr_url):
            st.session_state.pr_url = pr_url.rstrip("/")
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.interrupt_payload = None
            st.session_state.final = None
            st.session_state.unfinished = True
            st.query_params["thread"] = st.session_state.thread_id
            advance()
        else:
            st.error("Enter a URL such as https://github.com/owner/repo/pull/1.")

    payload = st.session_state.interrupt_payload
    if payload:
        if payload["kind"] == "approval_request":
            answer = render_approval_card(payload)
        elif payload["kind"] == "escalation":
            answer = render_escalation_card(payload)
        else:
            st.error(f"Unknown interrupt: {payload['kind']}")
            answer = None
        if answer is not None:
            advance(answer)
    elif st.session_state.unfinished:
        st.warning("This session stopped before completion. Continue from its saved checkpoint.")
        if st.button("Continue session"):
            advance()

    final = st.session_state.final
    if final is not None:
        action = final.get("final_action")
        if action in {"auto_approved", "committed"}:
            st.success("Review comment posted to GitHub.")
            analysis = final.get("analysis")
            if analysis:
                st.metric("Confidence", f"{analysis.confidence:.0%}")
                st.markdown(analysis.summary)
            st.link_button("View comment on GitHub", final.get("posted_comment_url") or st.session_state.pr_url)
            with st.expander("Posted review"):
                st.markdown(final.get("posted_comment_body") or "")
        elif action == "rejected":
            st.warning("Rejected — no comment posted.")
        elif action == "commit_failed":
            st.error(f"Posting the review failed: {final.get('commit_error') or 'Check GitHub credentials and permissions.'}")
        else:
            st.info(f"Session status: {action or 'in progress'}")
    if st.session_state.thread_id:
        st.caption(f"Session: {st.session_state.thread_id}")


if __name__ == "__main__":
    main()
