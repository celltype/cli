"""Tests for research-ops tools (notebook/todo/workflow templates)."""

from __future__ import annotations

from pathlib import Path

from ct.tools.ops import (
    notebook_add,
    notebook_delete,
    notebook_get,
    notebook_list,
    notebook_search,
    notebook_update,
    todo_add,
    todo_delete,
    todo_get,
    todo_list,
    todo_update,
    workflow_delete,
    workflow_get,
    workflow_list,
    workflow_run,
    workflow_save,
    workflow_update,
)


class _StubConfig:
    def __init__(self, base_dir: Path):
        self._base_dir = base_dir

    def get(self, key: str, default=None):
        if key == "ops.base_dir":
            return str(self._base_dir)
        return default


class _StubSession:
    def __init__(self, base_dir: Path):
        self.config = _StubConfig(base_dir)


def test_notebook_add_and_search(tmp_path):
    session = _StubSession(tmp_path)

    first = notebook_add(
        title="TP53 lead rationale",
        content="Prioritize TP53 synthetic lethal mapping in AML models.",
        tags=["TP53", "AML"],
        linked_query="Map TP53 vulnerabilities in AML",
        _session=session,
    )
    assert "error" not in first

    second = notebook_add(
        title="Immune caveat",
        content="Immune-cold phenotype suggests IO combo follow-up.",
        tags="immuno, aml",
        _session=session,
    )
    assert "error" not in second

    by_query = notebook_search(query="immune-cold", _session=session)
    assert by_query["count"] == 1
    assert by_query["matches"][0]["title"] == "Immune caveat"

    by_tag = notebook_search(tag="aml", _session=session)
    assert by_tag["count"] == 2


def test_notebook_search_tolerates_malformed_lines(tmp_path):
    session = _StubSession(tmp_path)
    notebook_add(title="ok", content="entry", _session=session)

    notebook_file = tmp_path / "notebook.jsonl"
    with open(notebook_file, "a", encoding="utf-8") as handle:
        handle.write("{bad json line}\n")

    result = notebook_search(query="entry", _session=session)
    assert result["count"] == 1
    assert result["skipped_malformed_lines"] == 1


def test_notebook_search_rejects_invalid_tag(tmp_path):
    session = _StubSession(tmp_path)
    result = notebook_search(tag="!!!!", _session=session)
    assert result["error"] == "invalid_tag"


def test_notebook_crud_cycle(tmp_path):
    session = _StubSession(tmp_path)
    created = notebook_add(title="Initial", content="alpha", tags="x,y", _session=session)
    eid = created["entry"]["id"]

    got = notebook_get(entry_id=eid, _session=session)
    assert got["title"] == "Initial"

    updated = notebook_update(entry_id=eid, title="Updated", content="beta", tags=["z"], _session=session)
    assert updated["entry"]["title"] == "Updated"
    assert updated["entry"]["tags"] == ["z"]

    listing = notebook_list(limit=5, _session=session)
    assert listing["count"] == 1
    assert listing["entries"][0]["id"] == eid

    deleted = notebook_delete(entry_id=eid, _session=session)
    assert "error" not in deleted
    missing = notebook_get(entry_id=eid, _session=session)
    assert missing["error"] == "not_found"


def test_todo_add_validates_due_date(tmp_path):
    session = _StubSession(tmp_path)

    result = todo_add(task="Run orthogonal assay", due_date="2026/01/01", _session=session)
    assert result["error"] == "invalid_due_date"


def test_todo_list_orders_by_priority(tmp_path):
    session = _StubSession(tmp_path)

    todo_add(task="Medium task", priority="medium", _session=session)
    todo_add(task="Critical task", priority="critical", _session=session)
    todo_add(task="High task", priority="high", _session=session)

    listing = todo_list(status="open", _session=session)
    assert listing["count"] == 3
    names = [item["task"] for item in listing["items"]]
    assert names[0] == "Critical task"
    assert names[1] == "High task"


def test_todo_crud_cycle(tmp_path):
    session = _StubSession(tmp_path)
    created = todo_add(task="Do something", priority="medium", _session=session)
    tid = created["item"]["id"]

    loaded = todo_get(todo_id=tid, _session=session)
    assert loaded["task"] == "Do something"

    updated = todo_update(todo_id=tid, status="in_progress", priority="high", owner="alice", _session=session)
    assert updated["item"]["status"] == "in_progress"
    assert updated["item"]["priority"] == "high"
    assert updated["item"]["owner"] == "alice"

    deleted = todo_delete(todo_id=tid, _session=session)
    assert "error" not in deleted
    missing = todo_get(todo_id=tid, _session=session)
    assert missing["error"] == "not_found"


def test_workflow_save_requires_valid_steps(tmp_path):
    session = _StubSession(tmp_path)

    bad = workflow_save(name="Bad", query="x", steps="not-json", _session=session)
    assert bad["error"] == "invalid_steps"

    missing_tool = workflow_save(
        name="Bad2",
        query="x",
        steps=[{"description": "no tool"}],
        _session=session,
    )
    assert missing_tool["error"] == "invalid_steps"


def test_workflow_save_creates_unique_files(tmp_path):
    session = _StubSession(tmp_path)
    steps = [{"id": 1, "description": "search", "tool": "literature.pubmed_search", "tool_args": {"query": "TP53"}}]

    first = workflow_save(name="TP53 Discovery", query="TP53 evidence", steps=steps, _session=session)
    second = workflow_save(name="TP53 Discovery", query="TP53 evidence", steps=steps, _session=session)

    assert "error" not in first
    assert "error" not in second
    assert first["path"] != second["path"]
    assert Path(first["path"]).exists()
    assert Path(second["path"]).exists()


def test_workflow_crud_and_run(tmp_path):
    session = _StubSession(tmp_path)
    steps = [
        {"id": 1, "description": "note", "tool": "ops.notebook_add", "tool_args": {"title": "w1", "content": "c1"}},
        {"id": 2, "description": "search", "tool": "ops.notebook_search", "tool_args": {"query": "$step.1.entry.title"}},
    ]
    saved = workflow_save(name="OpsWorkflow", query="demo", steps=steps, _session=session)
    wid = Path(saved["path"]).stem

    listed = workflow_list(_session=session)
    assert listed["count"] == 1
    assert listed["workflows"][0]["id"] == wid

    loaded = workflow_get(workflow_id=wid, _session=session)
    assert loaded["name"] == "OpsWorkflow"

    updated = workflow_update(workflow_id=wid, notes="updated", _session=session)
    assert updated["workflow"]["notes"] == "updated"

    dry = workflow_run(workflow_id=wid, dry_run=True, _session=session)
    assert dry["workflow_id"] == wid
    assert len(dry["steps"]) == 2

    ran = workflow_run(workflow_id=wid, _session=session)
    assert ran["status_by_step"][1] == "completed"
    assert ran["status_by_step"][2] == "completed"

    deleted = workflow_delete(workflow_id=wid, _session=session)
    assert "error" not in deleted
