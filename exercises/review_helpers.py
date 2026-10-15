"""Review presentation and validation shared by the CLI and approval UI."""

from common.schemas import ReviewState


def post_review_with_url(pr_url: str, body: str) -> str:
    """Use the lab's GitHub client settings and retain the created comment URL."""
    import httpx
    from common.github import API, _headers, parse_pr_url

    owner, repo, number = parse_pr_url(pr_url)
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{API}/repos/{owner}/{repo}/issues/{number}/comments",
            headers=_headers(), json={"body": body},
        )
        response.raise_for_status()
        return response.json()["html_url"]


def render_review(state: ReviewState) -> str:
    analysis = state["analysis"]
    lines = [
        f"### Automated review (confidence {analysis.confidence:.0%})",
        "", analysis.summary, "",
    ]
    for comment in analysis.comments:
        lines.append(
            f"- **[{comment.severity}]** `{comment.file}:{comment.line or '?'}` — {comment.body}"
        )
    return "\n".join(lines)


def approval_response(response: dict) -> dict:
    if not isinstance(response, dict) or response.get("choice") not in {"approve", "reject", "edit"}:
        raise ValueError("Choose approve, reject, or edit.")
    if response["choice"] == "edit" and not (response.get("feedback") or "").strip():
        raise ValueError("An edited review body is required for Edit.")
    return response


def escalation_questions(state: ReviewState) -> list[str]:
    analysis = state["analysis"]
    if analysis.escalation_questions:
        return analysis.escalation_questions
    # Still ask concrete questions if the structured model omits them.
    if analysis.comments:
        return [
            f"In {c.file}:{c.line or '?'}, what context or mitigation addresses: {c.body}?"
            for c in analysis.comments[:4]
        ]
    files = ", ".join(state.get("pr_files", [])) or state.get("pr_title", "this PR")
    risks = "; ".join(analysis.risk_factors) or analysis.confidence_reasoning
    return [f"For {files}, what intent, tests, or mitigations address these concerns: {risks}?"]


def escalation_response(response: dict, questions: list[str]) -> dict[str, str]:
    if not isinstance(response, dict) or any(
        not isinstance(response.get(q), str) or not response[q].strip() for q in questions
    ):
        raise ValueError("Please answer every escalation question.")
    return {q: response[q].strip() for q in questions}
