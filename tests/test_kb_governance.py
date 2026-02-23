"""Tests for enterprise governance layer."""

from types import SimpleNamespace

from ct.agent.config import Config
from ct.agent.types import Plan, Step
from ct.kb.governance import GovernanceEngine


def _session_with_config(data: dict):
    cfg = Config(data=data)
    usage = SimpleNamespace(total_cost=0.0)
    llm = SimpleNamespace(usage=usage)
    return SimpleNamespace(config=cfg, get_llm=lambda: llm)


def test_check_tool_blocked_by_policy(tmp_path):
    session = _session_with_config(
        {
            "enterprise.enforce_policy": True,
            "enterprise.blocked_tools": "shell.run",
            "enterprise.audit_enabled": True,
            "enterprise.audit_dir": str(tmp_path / "audit"),
        }
    )
    gov = GovernanceEngine(session, session_id="sess1")
    allowed, reason = gov.check_tool("shell.run")
    assert not allowed
    assert "blocked" in reason.lower()


def test_apply_plan_policy_marks_blocked_steps(tmp_path):
    session = _session_with_config(
        {
            "enterprise.enforce_policy": True,
            "enterprise.blocked_tools": "shell.run",
            "enterprise.audit_enabled": True,
            "enterprise.audit_dir": str(tmp_path / "audit"),
        }
    )
    gov = GovernanceEngine(session, session_id="sess2")
    plan = Plan(
        query="q",
        steps=[
            Step(id=1, description="safe", tool="literature.pubmed_search"),
            Step(id=2, description="blocked", tool="shell.run"),
        ],
    )
    result = gov.apply_plan_policy(plan)
    assert result["blocked_count"] == 1
    assert plan.steps[1].status == "failed"
    assert plan.steps[1].result["error"] == "blocked_by_policy"


def test_audit_events_written(tmp_path):
    session = _session_with_config(
        {
            "enterprise.enforce_policy": False,
            "enterprise.audit_enabled": True,
            "enterprise.audit_dir": str(tmp_path / "audit"),
        }
    )
    gov = GovernanceEngine(session, session_id="sess3")
    gov.query_start(query="q", context={"a": 1})
    gov.query_end(duration_s=1.2, iterations=1, total_steps=2)
    log = tmp_path / "audit" / "sess3.audit.jsonl"
    assert log.exists()
    lines = log.read_text().strip().splitlines()
    assert len(lines) >= 2
