"""
AgentLoop: wraps AgentRunner with trajectory persistence and clarification.

Provides the ``AgentLoop`` class used by the interactive terminal for
multi-turn sessions with memory, and ``ClarificationNeeded`` for requesting
additional input from the user.
"""

import logging
import uuid
from dataclasses import dataclass, field

from ct.agent.runner import AgentRunner
from ct.agent.trace_store import TraceStore
from ct.agent.trajectory import Trajectory

logger = logging.getLogger("ct.loop")


@dataclass
class Clarification:
    """A request for user clarification before executing a query."""
    question: str
    missing: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


class ClarificationNeeded(Exception):
    """Raised when the planner needs additional information."""

    def __init__(self, clarification: Clarification):
        self.clarification = clarification
        super().__init__(clarification.question)


class AgentLoop:
    """Multi-turn agent loop with trajectory memory.

    Wraps ``AgentRunner`` (SDK-based) and maintains a ``Trajectory``
    for multi-turn session context.
    """

    def __init__(self, session):
        self.session = session
        self.trajectory = Trajectory()
        session_id = str(uuid.uuid4())[:8]
        self.trace_store = TraceStore(session_id=session_id)
        self._runner = AgentRunner(
            session, trajectory=self.trajectory, trace_store=self.trace_store,
        )

    def run(self, query: str, context: dict | None = None):
        """Execute a query and record it in the trajectory."""
        result = self._runner.run(query, context)

        # Check for clarification request in result
        if result and result.raw_results:
            clar_data = result.raw_results.get("clarification")
            if isinstance(clar_data, dict) and clar_data.get("clarification_needed"):
                raise ClarificationNeeded(Clarification(
                    question=clar_data.get("question", "Could you clarify?"),
                    missing=clar_data.get("missing", []),
                    suggestions=clar_data.get("suggestions", []),
                ))

        # Record turn in trajectory
        if result:
            tools_used = []
            if result.plan:
                tools_used = [s.tool for s in result.plan.steps if s.tool]
            self.trajectory.add_turn(
                query=query,
                answer=result.summary or "",
                plan=result.plan,
            )

        return result

    @classmethod
    def resume(cls, session, session_id: str):
        """Resume a saved session by ID."""
        trajectory = Trajectory.load(session_id)
        loop = cls(session)
        loop.trajectory = trajectory
        # Reuse the same session ID for trace continuity
        loop.trace_store = TraceStore(session_id=session_id)
        loop._runner = AgentRunner(
            session, trajectory=trajectory, trace_store=loop.trace_store,
        )
        return loop

    @classmethod
    def resume_latest(cls, session):
        """Resume the most recent saved session."""
        sessions = Trajectory.list_sessions()
        if not sessions:
            raise FileNotFoundError("No saved sessions found.")
        latest = sessions[0]
        return cls.resume(session, latest["session_id"])
