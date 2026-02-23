"""
Trajectory: session memory across queries in interactive mode.

Records queries, answers, entities mentioned, and tools used so the planner
can reference prior context ("earlier you found X, now also check Y").
Supports persistence to JSONL for session resume.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Turn:
    """A single query-answer turn in a session."""
    query: str
    answer: str
    entities: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class Trajectory:
    """Session memory: records turns and provides context for the planner."""

    def __init__(self, max_turns: int = 20, session_id: str = None, title: str = None):
        self.turns: list[Turn] = []
        self.max_turns = max_turns
        self.session_id = session_id
        self.title = title
        self.created_at = time.time()

    def add_turn(self, query: str, answer: str, plan=None):
        """Record a completed turn with entity extraction."""
        entities = []
        tools_used = []
        if plan and hasattr(plan, "steps"):
            tools_used = [s.tool for s in plan.steps if s.status == "completed"]

        turn = Turn(
            query=query,
            answer=answer,
            entities=entities,
            tools_used=tools_used,
        )
        self.turns.append(turn)

        # Drop oldest turns if over limit
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def context_for_planner(self) -> str:
        """Format recent turns as context for the planner prompt."""
        if not self.turns:
            return ""

        recent = self.turns[-5:]
        lines = ["## Session Context (prior queries this session)", ""]
        for i, turn in enumerate(recent, 1):
            entities_str = ", ".join(turn.entities) if turn.entities else "none"
            tools_str = ", ".join(turn.tools_used) if turn.tools_used else "none"
            # Truncate answer to first 200 chars for context
            answer_preview = turn.answer[:200] + "..." if len(turn.answer) > 200 else turn.answer
            lines.append(f"**Turn {i}**: {turn.query}")
            lines.append(f"  Entities: {entities_str}")
            lines.append(f"  Tools: {tools_str}")
            lines.append(f"  Finding: {answer_preview}")
            lines.append("")

        all_entities = self.entities()
        if all_entities:
            lines.append(f"**All entities this session**: {', '.join(all_entities)}")
            lines.append("")

        lines.append(
            "Use this context to: reference prior findings, avoid repeating work, "
            "resolve ambiguous entity references (e.g., 'it', 'the compound')."
        )
        return "\n".join(lines)

    def entities(self) -> list[str]:
        """All unique entities mentioned across the session, preserving order."""
        seen = set()
        result = []
        for turn in self.turns:
            for entity in turn.entities:
                if entity not in seen:
                    seen.add(entity)
                    result.append(entity)
        return result

    def save(self, path: Path):
        """Persist trajectory to a JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            # First line: session metadata
            meta = {
                "type": "meta",
                "session_id": self.session_id,
                "title": self.title,
                "created_at": self.created_at,
                "n_turns": len(self.turns),
            }
            f.write(json.dumps(meta) + "\n")
            # Subsequent lines: turns
            for turn in self.turns:
                record = {"type": "turn", **asdict(turn)}
                f.write(json.dumps(record) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Trajectory":
        """Load a trajectory from a JSONL file."""
        trajectory = cls()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    import logging
                    logging.getLogger("ct.trajectory").warning(
                        "Skipping corrupted line in trajectory %s: %s", path, e,
                    )
                    continue
                if record.get("type") == "meta":
                    trajectory.session_id = record.get("session_id")
                    trajectory.title = record.get("title")
                    trajectory.created_at = record.get("created_at", 0)
                elif record.get("type") == "turn":
                    turn = Turn(
                        query=record.get("query", ""),
                        answer=record.get("answer", ""),
                        entities=record.get("entities", []),
                        tools_used=record.get("tools_used", []),
                        timestamp=record.get("timestamp", 0),
                    )
                    trajectory.turns.append(turn)
        return trajectory

    @staticmethod
    def sessions_dir() -> Path:
        """Return the directory where sessions are stored."""
        d = Path.home() / ".ct" / "sessions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def list_sessions(cls) -> list[dict]:
        """List all saved sessions, most recent first."""
        sessions_dir = cls.sessions_dir()
        sessions = []
        for path in sessions_dir.glob("*.jsonl"):
            try:
                with open(path) as f:
                    first_line = f.readline().strip()
                    if first_line:
                        meta = json.loads(first_line)
                        if meta.get("type") == "meta":
                            meta["path"] = str(path)
                            sessions.append(meta)
            except (json.JSONDecodeError, OSError):
                continue
        # Sort by created_at descending (most recent first)
        sessions.sort(key=lambda s: s.get("created_at", 0), reverse=True)
        return sessions
