"""Exercise 3 — Escalation branch with reviewer Q&A.

When confidence < 58%, the agent doesn't ask approve/reject — it asks specific
clarifying questions and then synthesizes a refined review from the answers.
"""

from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from exercises.review_helpers import approval_response, escalation_questions, escalation_response, render_review
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()


def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {"pr_title": pr.title, "pr_diff": pr.diff, "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha}


def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    llm = get_llm(temperature=0).with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis = llm.invoke([
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
    console.print(f"  [green]✓[/green] confidence={analysis.confidence:.0%}, {len(analysis.escalation_questions)} question(s)")
    return {"analysis": analysis}


def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD: decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:    decision = "escalate"
    else:                           decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    return {"decision": decision}


def node_escalate(state: ReviewState) -> dict:
    """Ask the reviewer specific questions; return their answers in state."""
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
    return {"escalation_answers": escalation_response(answers, questions)}


def node_synthesize(state: ReviewState) -> dict:
    """Re-prompt LLM with the reviewer's answers and produce a refined review."""
    console.print("[cyan]→ synthesize[/cyan]")
    initial = state["analysis"]
    qa = "\n".join(f"Q: {q}\nA: {ans}" for q, ans in (state.get("escalation_answers") or {}).items())
    llm = get_llm(temperature=0).with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining the review with reviewer answers...[/dim]"):
        refined = llm.invoke([
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
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%} (was {initial.confidence:.0%})")
    return {"analysis": refined}


def node_human_approval(state):
    a = state["analysis"]
    response = interrupt({
        "kind": "approval_request", "pr_url": state["pr_url"],
        "confidence": a.confidence, "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"],
        "review_body": render_review(state),
    })
    response = approval_response(response)
    return {"human_choice": response["choice"], "human_feedback": response.get("feedback")}


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


def _post(state, label: str) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return label
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed"


def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    if state.get("human_choice") in {"approve", "edit"}:
        label = "committed_after_escalation" if state.get("escalation_answers") else "committed"
        return {"final_action": _post(state, label)}
    console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
    return {"final_action": "rejected"}


def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    return {"final_action": _post(state, "auto_approved")}


def build_graph():
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "human_approval")
    return g.compile(checkpointer=MemorySaver())


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Approve? conf={payload['confidence']:.0%}",
            border_style="green",
        ))
        console.print(payload["confidence_reasoning"], markup=False)
        console.print(payload.get("review_body", payload.get("comments", [])), markup=False)
        console.print(payload.get("diff_preview", ""), markup=False)
        choice = ""
        while choice not in {"approve", "reject", "edit"}:
            choice = console.input("approve/reject/edit? ").strip().lower()
        feedback = console.input("Edited review body: " if choice == "edit" else "Feedback: ").strip()
        while choice == "edit" and not feedback:
            feedback = console.input("Edited review body (required): ").strip()
        return {"choice": choice, "feedback": feedback}
    if kind == "escalation":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Escalation conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        console.print(payload["confidence_reasoning"], markup=False)
        console.print(payload.get("risk_factors", []))
        answers = {}
        for question in payload["questions"]:
            answer = ""
            while not answer:
                answer = console.input(f"Q: {question}\nA: ").strip()
            answers[question] = answer
        return answers
    raise ValueError(kind)


def main():
    load_dotenv()
    p = argparse.ArgumentParser(); p.add_argument("--pr", required=True)
    args = p.parse_args()

    console.rule("[bold]Exercise 3 — escalation with reviewer Q&A[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    result = app.invoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume=handle_interrupt(result["__interrupt__"][0].value)), cfg)

    console.rule("Final")
    console.print(f"final_action = {result.get('final_action')}")
    if "analysis" in result:
        console.print(f"final confidence = {result['analysis'].confidence:.0%}")


if __name__ == "__main__":
    main()
