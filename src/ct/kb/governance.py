"""
Enterprise governance layer: policy enforcement + audit logging.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any


def _parse_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in str(value).split(",") if item.strip()}


class GovernanceEngine:
    """Evaluates runtime policy and writes audit events."""

    def __init__(self, session, *, session_id: str):
        self.session = session
        self.session_id = session_id
        self.audit_enabled = bool(session.config.get("enterprise.audit_enabled", True))
        self.enforce_policy = bool(session.config.get("enterprise.enforce_policy", False))
        audit_dir = Path(session.config.get("enterprise.audit_dir", str(Path.home() / ".ct" / "audit")))
        self.audit_path = audit_dir / f"{session_id}.audit.jsonl"

    def check_tool(self, tool_name: str) -> tuple[bool, str]:
        """Return whether tool execution is allowed under active policy."""
        if not self.enforce_policy:
            return True, ""

        blocked_tools = _parse_csv(self.session.config.get("enterprise.blocked_tools", ""))
        blocked_categories = _parse_csv(self.session.config.get("enterprise.blocked_categories", ""))
        require_allow = bool(self.session.config.get("enterprise.require_tool_allowlist", False))
        allowlist = _parse_csv(self.session.config.get("enterprise.tool_allowlist", ""))

        category = tool_name.split(".", 1)[0] if "." in tool_name else tool_name
        if tool_name in blocked_tools:
            return False, f"Tool blocked by policy: {tool_name}"
        if category in blocked_categories:
            return False, f"Tool category blocked by policy: {category}"
        if require_allow and tool_name not in allowlist:
            return False, f"Tool not in enterprise allowlist: {tool_name}"
        return True, ""

    def apply_plan_policy(self, plan) -> dict[str, Any]:
        """Pre-flight policy validation for plan steps."""
        blocked = []
        for step in getattr(plan, "steps", []):
            allowed, reason = self.check_tool(step.tool)
            if allowed:
                continue
            step.status = "failed"
            step.result = {"error": "blocked_by_policy", "summary": reason}
            blocked.append({"step_id": step.id, "tool": step.tool, "reason": reason})
        if blocked:
            self.audit_event("plan_policy_block", {"blocked_steps": blocked})
        return {"blocked_steps": blocked, "blocked_count": len(blocked)}

    def query_start(self, *, query: str, context: dict[str, Any] | None = None):
        self.audit_event(
            "query_start",
            {
                "query": query,
                "context_keys": sorted((context or {}).keys()),
                "profile": self.session.config.get("agent.profile", "research"),
            },
        )

    def query_end(self, *, duration_s: float, iterations: int, total_steps: int):
        max_cost = float(self.session.config.get("enterprise.max_cost_usd_per_query", 0.0) or 0.0)
        llm = self.session.get_llm()
        actual_cost = float(getattr(getattr(llm, "usage", None), "total_cost", 0.0) or 0.0)
        exceeded_cost_budget = bool(max_cost > 0 and actual_cost > max_cost)
        self.audit_event(
            "query_end",
            {
                "duration_s": round(duration_s, 4),
                "iterations": iterations,
                "total_steps": total_steps,
                "llm_cost_usd": round(actual_cost, 6),
                "cost_budget_usd": max_cost,
                "cost_budget_exceeded": exceeded_cost_budget,
            },
        )

    def audit_event(self, event_type: str, payload: dict[str, Any]):
        """Append an audit event."""
        if not self.audit_enabled:
            return
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            event = {
                "timestamp": time.time(),
                "session_id": self.session_id,
                "event_type": event_type,
                "payload": payload,
            }
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except OSError:
            # Audit logging is best-effort; policy checks still run.
            return
