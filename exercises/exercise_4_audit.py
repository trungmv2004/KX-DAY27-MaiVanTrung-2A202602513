"""Exercise 4 — Structured SQLite audit trail + durable checkpointer.

Zero setup — SQLite stores everything in a single file (`./hitl_audit.db`).
The audit_events schema is created automatically on first connection.

Goals:

1. Use AsyncSqliteSaver so the graph can resume after a crash.
2. Define and emit an `AuditEntry` (common/schemas.py) for every meaningful step,
   so the full session can be replayed.
3. Verify with `uv run python -m audit.replay --thread <id>`.

Approach:
    - Read `node_fetch_pr` below — it is the one fully-worked example. Pay attention
      to *why* each AuditEntry field has the value it does at that step.
    - For every other node, you decide what to log. Field reference:
      `common/schemas.py:AuditEntry`. Helper: `risk_level_for(confidence)`.
    - Implement the one-line body of `audit()`. Everything else (graph wiring,
      checkpointer setup, interrupt/resume loop) is already done for you.

Approval and escalation requests have separate checkpointed nodes so pending
audit events are not written again when interrupt() replays its node.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console

from common.db import db_path, write_audit_event
from common.github import fetch_pr
from common.llm import get_llm
from exercises.review_helpers import (
    approval_response, escalation_questions, escalation_response, render_review,
    post_review_with_url as post_review_comment,
)
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)


console = Console()
AGENT_ID = "pr-review-agent@v0.1"


class AuditedReviewState(ReviewState, total=False):
    posted_comment_url: str | None
    commit_error: str | None


async def audit(state, entry: AuditEntry) -> None:
    """Write one structured AuditEntry row to the `audit_events` table.

    `thread_id` and `pr_url` are taken from `state` so callers only build
    the entry itself.
    """
    await write_audit_event(thread_id=state["thread_id"], pr_url=state["pr_url"], entry=entry)


# ─── Reference example — read this carefully ───────────────────────────────
async def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = await asyncio.to_thread(fetch_pr, state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    # We've only fetched the diff, not analyzed it. So:
    #   - confidence is unknown → 0.0
    #   - risk_level can't be derived from confidence yet → "med" as neutral default
    #   - decision is "pending" — nothing has been decided
    #   - reviewer_id is None — no human is involved at this stage
    #   - reason is a short human-readable summary of what happened
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="fetch_pr",
        confidence=0.0,
        risk_level="med",
        decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }
# ───────────────────────────────────────────────────────────────────────────


async def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    t0 = time.monotonic()
    llm = get_llm(temperature=0).with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        a: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": (
                "You are a senior software engineer reviewing a pull request. "
                "Summarize what the PR does, list risk factors, and propose concrete "
                "review comments (file, line when known, severity, body). "
                "Then report your confidence (0.0-1.0) that your review is complete "
                "and correct, and explain why in confidence_reasoning.\n\n"
                "Pick the confidence in two steps:\n"
                "1. Does the diff ADD a security vulnerability? Only these count: "
                "insecure password hashing (MD5/SHA1), secrets or tokens stored in "
                "plaintext, SQL built with string formatting/concatenation, hard-coded "
                "user ids or credentials, credentials sent over plain HTTP. "
                "Parameterized SQL, schema changes, missing input validation and "
                "missing docs are NOT security vulnerabilities.\n"
                "2. Choose the band:\n"
                "- vulnerability found → 0.35-0.55, mark those comments 'blocker'.\n"
                "- no vulnerability, but it changes a database schema or persisted "
                "data format, or leaves open design questions a human should "
                "confirm (e.g. migrating existing data) → 0.60-0.70.\n"
                "- no vulnerability, trivial and self-contained, with NO database "
                "schema or storage change → 0.75-0.90. A PR that edits a CREATE TABLE "
                "statement is never trivial: existing databases need a migration.\n\n"
                "If your confidence is below "
                f"{ESCALATE_THRESHOLD:.0%}, populate escalation_questions with 2-4 "
                "specific, context-rich questions for the PR author. Each question must "
                "name the file (and function or line) in the diff it is about and ask "
                "for the intent or missing context you need, e.g. 'In tasktrack/auth.py "
                "hash_password(), why MD5 instead of bcrypt/argon2?'. Do not ask "
                "generic questions. Otherwise leave escalation_questions empty."
            )},
            {"role": "user", "content": f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}"},
        ])
    console.print(f"  [green]✓[/green] confidence={a.confidence:.0%}, {len(a.comments)} comment(s)")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="analyze",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=a.model_dump_json(),
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": a}


async def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    t0 = time.monotonic()
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="route",
        confidence=c,
        risk_level=risk_level_for(c),
        decision={"auto_approve": "auto", "human_approval": "pending", "escalate": "escalate"}[decision],
        reason=f"Routed to {decision} (confidence={c:.0%}, "
               f"thresholds auto>={AUTO_APPROVE_THRESHOLD:.0%}, escalate<{ESCALATE_THRESHOLD:.0%})",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"decision": decision}


async def node_request_approval(state):
    t0 = time.monotonic()
    a = state["analysis"]

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason="Waiting for reviewer approve/reject/edit",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {}


async def node_human_approval(state):
    t0 = time.monotonic()
    a = state["analysis"]

    resp = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"],
        "review_body": render_review(state),
        "refined": bool(state.get("escalation_answers")),
    })
    resp = approval_response(resp)

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision=resp.get("choice") or "pending",
        reason=resp.get("feedback") or None,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"human_choice": resp.get("choice"), "human_feedback": resp.get("feedback")}


def _render_comment_body(state) -> str:
    if state.get("human_choice") == "edit":
        return state["human_feedback"].strip()
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state) -> dict:
    try:
        body = _render_comment_body(state)
        url = post_review_comment(state["pr_url"], body)
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return {
            "final_action": "committed", "posted_comment_body": body,
            "posted_comment_url": url, "commit_error": None,
        }
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return {
            "final_action": "commit_failed", "posted_comment_body": None,
            "posted_comment_url": None, "commit_error": str(e),
        }


async def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    t0 = time.monotonic()
    if state.get("human_choice") in {"approve", "edit"}:
        result = await asyncio.to_thread(_post, state)
    else:
        console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
        result = {"final_action": "rejected", "posted_comment_body": None, "posted_comment_url": None}
    action = result["final_action"]
    a = state["analysis"]
    decision = state.get("human_choice") or "pending"
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="commit",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision=decision,
        reason=f"final_action={action}; {result.get('commit_error') or result.get('posted_comment_url') or ''}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return result


async def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    t0 = time.monotonic()
    a = state["analysis"]
    result = await asyncio.to_thread(_post, state)
    final_action = "auto_approved" if result["final_action"] == "committed" else "commit_failed"
    result["final_action"] = final_action
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="auto_approve",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="auto",
        reason=f"final_action={final_action}; {result.get('commit_error') or result.get('posted_comment_url') or ''}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return result


async def node_request_escalation(state):
    t0 = time.monotonic()
    a = state["analysis"]
    questions = escalation_questions(state)

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason="Questions: " + "\n".join(questions),
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {}


async def node_escalate(state):
    t0 = time.monotonic()
    a = state["analysis"]
    questions = escalation_questions(state)

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })
    answers = escalation_response(answers, questions)

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision="escalate",
        reason="\n".join(f"Q: {q}\nA: {answer}" for q, answer in answers.items()),
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"escalation_answers": answers}


async def node_synthesize(state):
    console.print("[cyan]→ synthesize[/cyan]")
    t0 = time.monotonic()
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in (state.get("escalation_answers") or {}).items())
    initial = state["analysis"]
    llm = get_llm(temperature=0).with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": (
                "You are a senior software engineer finalizing a pull request review. "
                "You already produced an initial analysis but were not confident, so a "
                "human reviewer answered your clarifying questions. Rewrite the review "
                "using their answers: drop concerns the answers resolve, keep or sharpen "
                "the ones they confirm, and make each comment concrete and actionable. "
                "Update confidence and confidence_reasoning to reflect what you now know. "
                "Leave escalation_questions empty."
            )},
            {"role": "user", "content": (
                f"Title: {state['pr_title']}\n\nDiff:\n{state['pr_diff']}\n\n"
                f"Initial analysis:\n{initial.model_dump_json(indent=2)}\n\n"
                f"Reviewer Q&A:\n{qa}"
            )},
        ])
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="synthesize",
        confidence=refined.confidence,
        risk_level=risk_level_for(refined.confidence),
        decision="pending",
        reason=refined.model_dump_json(),
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": refined}


def build_graph(checkpointer):
    g = StateGraph(AuditedReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
        ("request_approval", node_request_approval),
        ("request_escalation", node_request_escalation),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "request_approval", "escalate": "request_escalation"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("request_approval", "human_approval")
    g.add_edge("request_escalation", "escalate")
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "request_approval")
    return g.compile(checkpointer=checkpointer)


def handle_interrupt(payload):
    from exercises.exercise_3_escalation import handle_interrupt as prompt
    return prompt(payload)


async def invoke_session(app, pr_url: str, thread_id: str, resume_value=None):
    """Start a new thread, inspect a pause, or continue a saved checkpoint."""
    cfg = {"configurable": {"thread_id": thread_id}}
    snapshot = await app.aget_state(cfg)
    if snapshot.values and snapshot.values.get("pr_url") != pr_url:
        raise ValueError("This thread belongs to a different PR.")
    interrupts = [i for task in snapshot.tasks for i in task.interrupts]
    if resume_value is not None:
        if not interrupts:
            raise ValueError("This session is not waiting for a reviewer.")
        return await app.ainvoke(Command(resume=resume_value), cfg)
    if interrupts:
        return {**snapshot.values, "__interrupt__": tuple(interrupts)}
    if snapshot.values:
        return await app.ainvoke(None, cfg) if snapshot.next else snapshot.values
    return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)


async def run(pr_url: str, thread_id: str | None):
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 — SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        result = await invoke_session(app, pr_url, thread_id)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await invoke_session(app, pr_url, thread_id, handle_interrupt(payload))

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--thread", help="Resume an existing thread")
    args = p.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()
