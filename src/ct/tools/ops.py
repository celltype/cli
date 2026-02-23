"""
Research-ops productivity tools: notebook entries, todos, and workflow templates.

These tools provide lightweight project memory in ~/.ct/ops (or config override)
without depending on external services.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from ct.tools import registry


def _ops_root(_session=None) -> Path:
    """Return base directory for ops storage and ensure it exists."""
    base = None
    if _session is not None and getattr(_session, "config", None) is not None:
        base = _session.config.get("ops.base_dir")

    root = Path(base).expanduser() if base else (Path.home() / ".ct" / "ops")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _notebook_path(_session=None) -> Path:
    return _ops_root(_session) / "notebook.jsonl"


def _todos_path(_session=None) -> Path:
    return _ops_root(_session) / "todos.json"


def _workflow_dir(_session=None) -> Path:
    path = _ops_root(_session) / "workflows"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_tags(tags: Any) -> list[str]:
    """Normalize tags from string/list input to unique lowercase tokens."""
    if tags is None:
        return []

    raw: list[str]
    if isinstance(tags, str):
        raw = [x.strip() for x in tags.split(",")]
    elif isinstance(tags, list):
        raw = [str(x).strip() for x in tags]
    else:
        raw = [str(tags).strip()]

    cleaned: list[str] = []
    seen = set()
    for tag in raw:
        if not tag:
            continue
        token = re.sub(r"\s+", "-", tag.lower())
        token = re.sub(r"[^a-z0-9._:-]", "", token)
        if not token or token in seen:
            continue
        seen.add(token)
        cleaned.append(token)

    return cleaned[:20]


def _load_todos(_session=None) -> tuple[list[dict], str | None]:
    """Load todo list, returning (todos, error)."""
    path = _todos_path(_session)
    if not path.exists():
        return [], None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], f"Failed to read todo database: {exc}"

    if not isinstance(data, list):
        return [], "Todo database is corrupted (expected JSON list)."

    todos = [x for x in data if isinstance(x, dict)]
    return todos, None


def _save_todos(todos: list[dict], _session=None) -> tuple[Path, str | None]:
    """Persist todo list to disk."""
    path = _todos_path(_session)
    try:
        path.write_text(json.dumps(todos, indent=2), encoding="utf-8")
        return path, None
    except Exception as exc:
        return path, f"Failed to persist todo database: {exc}"


def _priority_rank(priority: str) -> int:
    mapping = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return mapping.get(priority, 2)


def _slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-._")
    return slug or "workflow"


def _load_notebook_entries(_session=None) -> tuple[list[dict], int, str | None]:
    """Load notebook entries from JSONL, skipping malformed lines."""
    path = _notebook_path(_session)
    if not path.exists():
        return [], 0, None

    entries: list[dict] = []
    skipped = 0
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if isinstance(obj, dict):
                    entries.append(obj)
                else:
                    skipped += 1
    except Exception as exc:
        return [], 0, f"Failed to read notebook: {exc}"

    return entries, skipped, None


def _save_notebook_entries(entries: list[dict], _session=None) -> tuple[Path, str | None]:
    """Persist notebook entries to JSONL."""
    path = _notebook_path(_session)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry) + "\n")
        return path, None
    except Exception as exc:
        return path, f"Failed to persist notebook: {exc}"


def _list_workflow_files(_session=None) -> list[Path]:
    path = _workflow_dir(_session)
    return sorted([p for p in path.glob("*.json") if p.is_file()])


def _load_workflow_payload(path: Path) -> tuple[dict | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"Failed to read workflow file '{path.name}': {exc}"
    if not isinstance(payload, dict):
        return None, f"Invalid workflow file '{path.name}' (expected object)."
    return payload, None


def _find_workflow_path(identifier: str, _session=None) -> tuple[Path | None, str | None]:
    """Resolve workflow by file stem, file name, or display name."""
    ident = (identifier or "").strip()
    if not ident:
        return None, "Workflow identifier is required."

    candidates = _list_workflow_files(_session)
    if not candidates:
        return None, "No saved workflows."

    # 1) Exact filename match
    for path in candidates:
        if path.name == ident or path.stem == ident:
            return path, None

    # 2) Slug match
    slug = _slugify_name(ident)
    for path in candidates:
        if path.stem == slug:
            return path, None

    # 3) Match by workflow display name in payload
    lowered = ident.lower()
    for path in candidates:
        payload, error = _load_workflow_payload(path)
        if error or not payload:
            continue
        name = str(payload.get("name", "")).strip().lower()
        if name == lowered:
            return path, None

    return None, f"Workflow '{identifier}' not found."


@registry.register(
    name="ops.notebook_add",
    description="Append a structured notebook entry for project memory",
    category="ops",
    parameters={
        "title": "Short notebook entry title",
        "content": "Entry body text (markdown/plain text)",
        "tags": "Optional list of tags or comma-separated tags",
        "linked_query": "Optional source query/command that produced the insight",
    },
    usage_guide=(
        "Use after an important finding, decision, or caveat so future runs can reuse context. "
        "Prefer concise entries with tags for retrieval."
    ),
)
def notebook_add(
    title: str,
    content: str,
    tags: list[str] | str | None = None,
    linked_query: str | None = None,
    _session=None,
    **kwargs,
) -> dict:
    """Append a notebook entry to local JSONL storage."""
    title = (title or "").strip()
    content = (content or "").strip()
    if not title:
        return {"summary": "Notebook title is required.", "error": "missing_title"}
    if not content:
        return {"summary": "Notebook content is required.", "error": "missing_content"}

    entry = {
        "id": uuid4().hex[:12],
        "created_at": _now_iso(),
        "title": title,
        "content": content,
        "tags": _normalize_tags(tags),
        "linked_query": (linked_query or "").strip() or None,
    }

    path = _notebook_path(_session)
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except Exception as exc:
        return {"summary": f"Failed to save notebook entry: {exc}", "error": "write_failed"}

    return {
        "summary": f"Notebook entry saved ({entry['id']}) with {len(entry['tags'])} tags.",
        "entry": entry,
        "path": str(path),
    }


@registry.register(
    name="ops.notebook_search",
    description="Search notebook entries by keyword and/or tag",
    category="ops",
    parameters={
        "query": "Keyword query matched against title/content/linked_query",
        "tag": "Optional single tag filter",
        "limit": "Maximum entries to return (default 20, max 100)",
    },
    usage_guide=(
        "Use before planning to recover prior findings, assumptions, and unresolved risks. "
        "Combine with tags to narrow to specific projects."
    ),
)
def notebook_search(
    query: str = "",
    tag: str = "",
    limit: int = 20,
    _session=None,
    **kwargs,
) -> dict:
    """Search notebook JSONL entries."""
    limit = max(1, min(int(limit), 100))
    q = (query or "").strip().lower()
    tag_tokens = _normalize_tags([tag]) if tag else []
    tag_norm = tag_tokens[0] if tag_tokens else ""
    if tag and not tag_norm:
        return {
            "summary": "Invalid tag filter.",
            "error": "invalid_tag",
        }

    path = _notebook_path(_session)
    if not path.exists():
        return {
            "summary": f"Notebook is empty: {path}",
            "matches": [],
            "count": 0,
            "path": str(path),
        }

    matches = []
    bad_lines = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            if not isinstance(entry, dict):
                bad_lines += 1
                continue

            if tag_norm and tag_norm not in entry.get("tags", []):
                continue

            haystack = " ".join(
                [
                    str(entry.get("title", "")),
                    str(entry.get("content", "")),
                    str(entry.get("linked_query", "")),
                ]
            ).lower()
            if q and q not in haystack:
                continue

            preview = str(entry.get("content", "")).replace("\n", " ").strip()
            if len(preview) > 180:
                preview = preview[:177] + "..."

            matches.append(
                {
                    "id": entry.get("id"),
                    "created_at": entry.get("created_at"),
                    "title": entry.get("title"),
                    "tags": entry.get("tags", []),
                    "preview": preview,
                    "linked_query": entry.get("linked_query"),
                }
            )

    matches.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    matches = matches[:limit]

    qualifier = []
    if q:
        qualifier.append(f"query='{query}'")
    if tag_norm:
        qualifier.append(f"tag='{tag_norm}'")
    suffix = f" ({', '.join(qualifier)})" if qualifier else ""

    summary = f"Found {len(matches)} notebook entries{suffix}."
    if bad_lines:
        summary += f" Skipped {bad_lines} malformed lines."

    return {
        "summary": summary,
        "matches": matches,
        "count": len(matches),
        "path": str(path),
        "skipped_malformed_lines": bad_lines,
    }


@registry.register(
    name="ops.notebook_get",
    description="Fetch a notebook entry by ID",
    category="ops",
    parameters={"entry_id": "Notebook entry ID"},
    usage_guide="Use when you need the full text of one saved notebook entry.",
)
def notebook_get(entry_id: str, _session=None, **kwargs) -> dict:
    """Get a notebook entry by ID."""
    needle = (entry_id or "").strip()
    if not needle:
        return {"summary": "entry_id is required.", "error": "missing_entry_id"}

    entries, skipped, error = _load_notebook_entries(_session)
    if error:
        return {"summary": error, "error": "notebook_error"}

    for entry in entries:
        if str(entry.get("id", "")).strip() == needle:
            out = dict(entry)
            out["summary"] = f"Notebook entry {needle} loaded."
            if skipped:
                out["summary"] += f" Skipped {skipped} malformed lines while loading."
            out["path"] = str(_notebook_path(_session))
            return out

    return {"summary": f"Notebook entry not found: {needle}", "error": "not_found"}


@registry.register(
    name="ops.notebook_list",
    description="List recent notebook entries",
    category="ops",
    parameters={
        "limit": "Maximum entries to return (default 20, max 200)",
        "tag": "Optional tag filter",
    },
    usage_guide="Use for a quick overview of recent project notes.",
)
def notebook_list(limit: int = 20, tag: str = "", _session=None, **kwargs) -> dict:
    """List notebook entries sorted by recency."""
    limit = max(1, min(_parse_int(limit, 20), 200))
    tag_tokens = _normalize_tags([tag]) if tag else []
    tag_norm = tag_tokens[0] if tag_tokens else ""
    if tag and not tag_norm:
        return {"summary": "Invalid tag filter.", "error": "invalid_tag"}

    entries, skipped, error = _load_notebook_entries(_session)
    if error:
        return {"summary": error, "error": "notebook_error"}

    if tag_norm:
        entries = [e for e in entries if tag_norm in (e.get("tags") or [])]

    entries = sorted(entries, key=lambda x: str(x.get("created_at") or ""), reverse=True)
    entries = entries[:limit]

    items = []
    for entry in entries:
        preview = str(entry.get("content", "")).replace("\n", " ").strip()
        if len(preview) > 140:
            preview = preview[:137] + "..."
        items.append(
            {
                "id": entry.get("id"),
                "created_at": entry.get("created_at"),
                "title": entry.get("title"),
                "tags": entry.get("tags", []),
                "preview": preview,
            }
        )

    summary = f"Listed {len(items)} notebook entries."
    if tag_norm:
        summary += f" tag={tag_norm}."
    if skipped:
        summary += f" Skipped {skipped} malformed lines."

    return {
        "summary": summary,
        "entries": items,
        "count": len(items),
        "path": str(_notebook_path(_session)),
        "skipped_malformed_lines": skipped,
    }


@registry.register(
    name="ops.notebook_update",
    description="Update fields of an existing notebook entry",
    category="ops",
    parameters={
        "entry_id": "Notebook entry ID",
        "title": "Optional new title",
        "content": "Optional new content",
        "tags": "Optional replacement tags (list or comma-separated)",
        "linked_query": "Optional replacement linked query",
    },
    usage_guide="Use to correct or refine existing notes without creating duplicates.",
)
def notebook_update(
    entry_id: str,
    title: str | None = None,
    content: str | None = None,
    tags: list[str] | str | None = None,
    linked_query: str | None = None,
    _session=None,
    **kwargs,
) -> dict:
    """Update an existing notebook entry by ID."""
    needle = (entry_id or "").strip()
    if not needle:
        return {"summary": "entry_id is required.", "error": "missing_entry_id"}

    entries, skipped, error = _load_notebook_entries(_session)
    if error:
        return {"summary": error, "error": "notebook_error"}

    touched = None
    for entry in entries:
        if str(entry.get("id", "")).strip() != needle:
            continue
        if title is not None:
            entry["title"] = str(title).strip()
        if content is not None:
            entry["content"] = str(content).strip()
        if tags is not None:
            entry["tags"] = _normalize_tags(tags)
        if linked_query is not None:
            entry["linked_query"] = str(linked_query).strip() or None
        entry["updated_at"] = _now_iso()
        touched = entry
        break

    if touched is None:
        return {"summary": f"Notebook entry not found: {needle}", "error": "not_found"}
    if not str(touched.get("title", "")).strip():
        return {"summary": "Notebook title cannot be empty.", "error": "invalid_title"}
    if not str(touched.get("content", "")).strip():
        return {"summary": "Notebook content cannot be empty.", "error": "invalid_content"}

    path, error = _save_notebook_entries(entries, _session)
    if error:
        return {"summary": error, "error": "notebook_error"}

    summary = f"Notebook entry {needle} updated."
    if skipped:
        summary += f" Skipped {skipped} malformed lines while loading."
    return {"summary": summary, "entry": touched, "path": str(path)}


@registry.register(
    name="ops.notebook_delete",
    description="Delete a notebook entry by ID",
    category="ops",
    parameters={"entry_id": "Notebook entry ID"},
    usage_guide="Use to remove stale or incorrect notebook entries.",
)
def notebook_delete(entry_id: str, _session=None, **kwargs) -> dict:
    """Delete a notebook entry by ID."""
    needle = (entry_id or "").strip()
    if not needle:
        return {"summary": "entry_id is required.", "error": "missing_entry_id"}

    entries, skipped, error = _load_notebook_entries(_session)
    if error:
        return {"summary": error, "error": "notebook_error"}

    original = len(entries)
    kept = [e for e in entries if str(e.get("id", "")).strip() != needle]
    if len(kept) == original:
        return {"summary": f"Notebook entry not found: {needle}", "error": "not_found"}

    path, error = _save_notebook_entries(kept, _session)
    if error:
        return {"summary": error, "error": "notebook_error"}

    summary = f"Notebook entry deleted: {needle}."
    if skipped:
        summary += f" Skipped {skipped} malformed lines while loading."
    return {"summary": summary, "path": str(path), "count": len(kept)}


@registry.register(
    name="ops.todo_add",
    description="Create a tracked todo item for research follow-ups",
    category="ops",
    parameters={
        "task": "Todo description",
        "priority": "critical|high|medium|low (default medium)",
        "due_date": "Optional due date in YYYY-MM-DD",
        "owner": "Optional owner name/alias",
    },
    usage_guide=(
        "Use to capture follow-up actions from synthesis outputs (validation assays, "
        "data pulls, literature checks) so nothing is lost between sessions."
    ),
)
def todo_add(
    task: str,
    priority: str = "medium",
    due_date: str | None = None,
    owner: str | None = None,
    _session=None,
    **kwargs,
) -> dict:
    """Append a todo item to local todo storage."""
    task = (task or "").strip()
    if not task:
        return {"summary": "Todo task is required.", "error": "missing_task"}

    normalized_priority = (priority or "medium").strip().lower()
    allowed_priorities = {"critical", "high", "medium", "low"}
    if normalized_priority not in allowed_priorities:
        return {
            "summary": "Invalid priority. Use one of: critical, high, medium, low.",
            "error": "invalid_priority",
        }

    normalized_due = None
    if due_date:
        try:
            normalized_due = date.fromisoformat(str(due_date)).isoformat()
        except ValueError:
            return {
                "summary": "Invalid due_date format. Use YYYY-MM-DD.",
                "error": "invalid_due_date",
            }

    todos, err = _load_todos(_session)
    if err:
        return {"summary": err, "error": "todo_db_error"}

    item = {
        "id": uuid4().hex[:12],
        "task": task,
        "status": "open",
        "priority": normalized_priority,
        "due_date": normalized_due,
        "owner": (owner or "").strip() or None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    todos.append(item)

    path, err = _save_todos(todos, _session)
    if err:
        return {"summary": err, "error": "todo_db_error"}

    open_count = sum(1 for x in todos if x.get("status") == "open")
    return {
        "summary": f"Todo added ({item['id']}). Open items: {open_count}.",
        "item": item,
        "open_count": open_count,
        "path": str(path),
    }


@registry.register(
    name="ops.todo_list",
    description="List tracked todo items with status and priority ordering",
    category="ops",
    parameters={
        "status": "open|done|all (default open)",
        "limit": "Maximum items to return (default 50, max 200)",
    },
    usage_guide=(
        "Use at the start/end of sessions to manage execution backlog. "
        "Default ordering surfaces urgent and overdue items first."
    ),
)
def todo_list(status: str = "open", limit: int = 50, _session=None, **kwargs) -> dict:
    """Return todo items with deterministic ordering."""
    status_norm = (status or "open").strip().lower()
    if status_norm not in {"open", "done", "all"}:
        return {
            "summary": "Invalid status. Use open, done, or all.",
            "error": "invalid_status",
        }

    limit = max(1, min(int(limit), 200))
    todos, err = _load_todos(_session)
    if err:
        return {"summary": err, "error": "todo_db_error"}

    filtered = todos
    if status_norm != "all":
        filtered = [x for x in todos if x.get("status") == status_norm]

    def sort_key(item: dict) -> tuple:
        due = item.get("due_date") or "9999-12-31"
        created = item.get("created_at") or ""
        return (_priority_rank(str(item.get("priority", "medium"))), due, created)

    filtered = sorted(filtered, key=sort_key)
    limited = filtered[:limit]

    open_count = sum(1 for x in todos if x.get("status") == "open")
    done_count = sum(1 for x in todos if x.get("status") == "done")

    return {
        "summary": f"Listed {len(limited)} todo items (open={open_count}, done={done_count}).",
        "items": limited,
        "count": len(limited),
        "open_count": open_count,
        "done_count": done_count,
        "status_filter": status_norm,
        "path": str(_todos_path(_session)),
    }


@registry.register(
    name="ops.todo_get",
    description="Fetch a todo item by ID",
    category="ops",
    parameters={"todo_id": "Todo item ID"},
    usage_guide="Use to inspect a single todo in full detail.",
)
def todo_get(todo_id: str, _session=None, **kwargs) -> dict:
    """Get a todo item by ID."""
    needle = (todo_id or "").strip()
    if not needle:
        return {"summary": "todo_id is required.", "error": "missing_todo_id"}

    todos, err = _load_todos(_session)
    if err:
        return {"summary": err, "error": "todo_db_error"}

    for item in todos:
        if str(item.get("id", "")).strip() == needle:
            out = dict(item)
            out["summary"] = f"Todo item loaded: {needle}"
            out["path"] = str(_todos_path(_session))
            return out
    return {"summary": f"Todo item not found: {needle}", "error": "not_found"}


@registry.register(
    name="ops.todo_update",
    description="Update an existing todo item",
    category="ops",
    parameters={
        "todo_id": "Todo item ID",
        "task": "Optional replacement task text",
        "status": "Optional status: open|in_progress|blocked|done|cancelled",
        "priority": "Optional priority: critical|high|medium|low",
        "due_date": "Optional due date in YYYY-MM-DD (or empty to clear)",
        "owner": "Optional owner (or empty to clear)",
    },
    usage_guide="Use to track execution state and ownership of follow-up work.",
)
def todo_update(
    todo_id: str,
    task: str | None = None,
    status: str | None = None,
    priority: str | None = None,
    due_date: str | None = None,
    owner: str | None = None,
    _session=None,
    **kwargs,
) -> dict:
    """Update a todo item by ID."""
    needle = (todo_id or "").strip()
    if not needle:
        return {"summary": "todo_id is required.", "error": "missing_todo_id"}

    todos, err = _load_todos(_session)
    if err:
        return {"summary": err, "error": "todo_db_error"}

    allowed_status = {"open", "in_progress", "blocked", "done", "cancelled"}
    allowed_priority = {"critical", "high", "medium", "low"}
    item = None
    for candidate in todos:
        if str(candidate.get("id", "")).strip() == needle:
            item = candidate
            break
    if item is None:
        return {"summary": f"Todo item not found: {needle}", "error": "not_found"}

    if task is not None:
        item["task"] = str(task).strip()
    if status is not None:
        normalized_status = str(status).strip().lower()
        if normalized_status not in allowed_status:
            return {
                "summary": "Invalid status. Use open, in_progress, blocked, done, cancelled.",
                "error": "invalid_status",
            }
        item["status"] = normalized_status
    if priority is not None:
        normalized_priority = str(priority).strip().lower()
        if normalized_priority not in allowed_priority:
            return {
                "summary": "Invalid priority. Use critical, high, medium, low.",
                "error": "invalid_priority",
            }
        item["priority"] = normalized_priority
    if due_date is not None:
        raw_due = str(due_date).strip()
        if raw_due:
            try:
                item["due_date"] = date.fromisoformat(raw_due).isoformat()
            except ValueError:
                return {
                    "summary": "Invalid due_date format. Use YYYY-MM-DD.",
                    "error": "invalid_due_date",
                }
        else:
            item["due_date"] = None
    if owner is not None:
        item["owner"] = str(owner).strip() or None

    if not str(item.get("task", "")).strip():
        return {"summary": "Todo task cannot be empty.", "error": "invalid_task"}
    item["updated_at"] = _now_iso()

    path, err = _save_todos(todos, _session)
    if err:
        return {"summary": err, "error": "todo_db_error"}
    return {"summary": f"Todo item updated: {needle}", "item": item, "path": str(path)}


@registry.register(
    name="ops.todo_delete",
    description="Delete a todo item by ID",
    category="ops",
    parameters={"todo_id": "Todo item ID"},
    usage_guide="Use to remove obsolete todo items.",
)
def todo_delete(todo_id: str, _session=None, **kwargs) -> dict:
    """Delete a todo item by ID."""
    needle = (todo_id or "").strip()
    if not needle:
        return {"summary": "todo_id is required.", "error": "missing_todo_id"}

    todos, err = _load_todos(_session)
    if err:
        return {"summary": err, "error": "todo_db_error"}

    original = len(todos)
    kept = [x for x in todos if str(x.get("id", "")).strip() != needle]
    if len(kept) == original:
        return {"summary": f"Todo item not found: {needle}", "error": "not_found"}

    path, err = _save_todos(kept, _session)
    if err:
        return {"summary": err, "error": "todo_db_error"}
    return {"summary": f"Todo item deleted: {needle}", "count": len(kept), "path": str(path)}


def _normalize_workflow_steps(steps: list[dict] | str) -> tuple[list[dict] | None, str | None]:
    """Validate and normalize workflow step payloads."""
    if isinstance(steps, str):
        try:
            steps = json.loads(steps)
        except json.JSONDecodeError:
            return None, "Invalid steps payload. Provide JSON array or list of step objects."

    if not isinstance(steps, list) or not steps:
        return None, "Workflow steps must be a non-empty list."

    cleaned_steps = []
    for idx, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            return None, f"Step {idx} is not an object."
        tool = str(step.get("tool", "")).strip()
        description = str(step.get("description", "")).strip()
        if not tool:
            return None, f"Step {idx} is missing required field 'tool'."

        cleaned_steps.append(
            {
                "id": _parse_int(step.get("id", idx), idx),
                "description": description,
                "tool": tool,
                "tool_args": step.get("tool_args", {}) if isinstance(step.get("tool_args", {}), dict) else {},
                "depends_on": [
                    _parse_int(x, 0) for x in (step.get("depends_on", []) if isinstance(step.get("depends_on", []), list) else [])
                    if _parse_int(x, 0) > 0
                ],
            }
        )

    cleaned_steps.sort(key=lambda x: x["id"])
    return cleaned_steps, None


@registry.register(
    name="ops.workflow_save",
    description="Save a reusable workflow template from a plan-like step list",
    category="ops",
    parameters={
        "name": "Workflow template name",
        "query": "Original or canonical query this workflow answers",
        "steps": "List of step dicts (id/description/tool/tool_args/depends_on)",
        "notes": "Optional notes about assumptions or context",
    },
    usage_guide=(
        "Use after a successful run to preserve the strategy as a reusable template. "
        "Templates are stored locally and can be inspected with files.read_file."
    ),
)
def workflow_save(
    name: str,
    query: str,
    steps: list[dict] | str,
    notes: str = "",
    _session=None,
    **kwargs,
) -> dict:
    """Persist a validated workflow template to local JSON."""
    workflow_name = (name or "").strip()
    if not workflow_name:
        return {"summary": "Workflow name is required.", "error": "missing_name"}

    query = (query or "").strip()
    if not query:
        return {"summary": "Workflow query is required.", "error": "missing_query"}

    cleaned_steps, error = _normalize_workflow_steps(steps)
    if error:
        return {"summary": error, "error": "invalid_steps"}

    payload = {
        "name": workflow_name,
        "query": query,
        "notes": (notes or "").strip() or None,
        "created_at": _now_iso(),
        "version": 1,
        "steps": cleaned_steps,
    }

    out_dir = _workflow_dir(_session)
    stem = _slugify_name(workflow_name)
    out_path = out_dir / f"{stem}.json"
    suffix = 1
    while out_path.exists():
        suffix += 1
        out_path = out_dir / f"{stem}-{suffix}.json"

    try:
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        return {"summary": f"Failed to save workflow template: {exc}", "error": "write_failed"}

    return {
        "summary": f"Saved workflow template '{workflow_name}' with {len(cleaned_steps)} steps.",
        "workflow": payload,
        "path": str(out_path),
    }


@registry.register(
    name="ops.workflow_list",
    description="List saved workflow templates",
    category="ops",
    parameters={
        "limit": "Maximum workflows to return (default 50, max 200)",
        "query": "Optional keyword filter against name/query/notes",
    },
    usage_guide="Use to discover reusable workflow templates available in local storage.",
)
def workflow_list(limit: int = 50, query: str = "", _session=None, **kwargs) -> dict:
    """List workflow template metadata."""
    limit = max(1, min(_parse_int(limit, 50), 200))
    needle = (query or "").strip().lower()
    workflows = []
    for path in _list_workflow_files(_session):
        payload, error = _load_workflow_payload(path)
        if error or not payload:
            continue
        haystack = " ".join(
            [
                str(payload.get("name", "")),
                str(payload.get("query", "")),
                str(payload.get("notes", "")),
            ]
        ).lower()
        if needle and needle not in haystack:
            continue
        workflows.append(
            {
                "id": path.stem,
                "name": payload.get("name", path.stem),
                "query": payload.get("query", ""),
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("updated_at"),
                "n_steps": len(payload.get("steps", []) if isinstance(payload.get("steps"), list) else []),
                "path": str(path),
            }
        )

    workflows.sort(key=lambda x: str(x.get("updated_at") or x.get("created_at") or ""), reverse=True)
    workflows = workflows[:limit]
    suffix = f" filter='{query}'" if needle else ""
    return {
        "summary": f"Listed {len(workflows)} workflow templates.{suffix}",
        "workflows": workflows,
        "count": len(workflows),
        "directory": str(_workflow_dir(_session)),
    }


@registry.register(
    name="ops.workflow_get",
    description="Load one saved workflow template by ID or name",
    category="ops",
    parameters={"workflow_id": "Workflow file stem, file name, or display name"},
    usage_guide="Use when you need full details of a saved workflow template.",
)
def workflow_get(workflow_id: str, _session=None, **kwargs) -> dict:
    """Get a workflow template payload."""
    path, error = _find_workflow_path(workflow_id, _session)
    if error:
        return {"summary": error, "error": "not_found"}

    payload, error = _load_workflow_payload(path)
    if error or payload is None:
        return {"summary": error or "Invalid workflow payload.", "error": "workflow_error"}

    payload = dict(payload)
    payload["summary"] = f"Workflow loaded: {payload.get('name', path.stem)}"
    payload["workflow_id"] = path.stem
    payload["path"] = str(path)
    return payload


@registry.register(
    name="ops.workflow_update",
    description="Update an existing workflow template",
    category="ops",
    parameters={
        "workflow_id": "Workflow file stem, file name, or display name",
        "name": "Optional replacement name",
        "query": "Optional replacement canonical query",
        "steps": "Optional replacement step list",
        "notes": "Optional replacement notes",
    },
    usage_guide="Use to keep reusable workflows current as your process evolves.",
)
def workflow_update(
    workflow_id: str,
    name: str | None = None,
    query: str | None = None,
    steps: list[dict] | str | None = None,
    notes: str | None = None,
    _session=None,
    **kwargs,
) -> dict:
    """Update workflow template fields and save in place."""
    path, error = _find_workflow_path(workflow_id, _session)
    if error:
        return {"summary": error, "error": "not_found"}

    payload, error = _load_workflow_payload(path)
    if error or payload is None:
        return {"summary": error or "Invalid workflow payload.", "error": "workflow_error"}

    if name is not None:
        payload["name"] = str(name).strip()
    if query is not None:
        payload["query"] = str(query).strip()
    if notes is not None:
        payload["notes"] = str(notes).strip() or None
    if steps is not None:
        cleaned_steps, step_error = _normalize_workflow_steps(steps)
        if step_error:
            return {"summary": step_error, "error": "invalid_steps"}
        payload["steps"] = cleaned_steps

    if not str(payload.get("name", "")).strip():
        return {"summary": "Workflow name cannot be empty.", "error": "invalid_name"}
    if not str(payload.get("query", "")).strip():
        return {"summary": "Workflow query cannot be empty.", "error": "invalid_query"}
    if not isinstance(payload.get("steps"), list) or not payload["steps"]:
        return {"summary": "Workflow requires at least one step.", "error": "invalid_steps"}

    payload["updated_at"] = _now_iso()
    payload["version"] = _parse_int(payload.get("version", 1), 1) + 1
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        return {"summary": f"Failed to update workflow: {exc}", "error": "write_failed"}

    return {
        "summary": f"Workflow updated: {payload.get('name', path.stem)}",
        "workflow": payload,
        "path": str(path),
    }


@registry.register(
    name="ops.workflow_delete",
    description="Delete a saved workflow template",
    category="ops",
    parameters={"workflow_id": "Workflow file stem, file name, or display name"},
    usage_guide="Use to remove obsolete workflow templates.",
)
def workflow_delete(workflow_id: str, _session=None, **kwargs) -> dict:
    """Delete one workflow template."""
    path, error = _find_workflow_path(workflow_id, _session)
    if error:
        return {"summary": error, "error": "not_found"}

    try:
        path.unlink()
    except Exception as exc:
        return {"summary": f"Failed to delete workflow: {exc}", "error": "delete_failed"}
    return {"summary": f"Workflow deleted: {path.stem}", "path": str(path)}


@registry.register(
    name="ops.workflow_run",
    description="Execute a saved workflow template",
    category="ops",
    parameters={
        "workflow_id": "Workflow file stem, file name, or display name",
        "dry_run": "If true, return the resolved execution plan without running tools",
        "continue_on_error": "If true, continue executing later steps after failures",
    },
    usage_guide="Use to replay a validated workflow template against new inputs or contexts.",
)
def workflow_run(
    workflow_id: str,
    dry_run: bool = False,
    continue_on_error: bool = False,
    _session=None,
    _prior_results=None,
    **kwargs,
) -> dict:
    """Execute workflow steps in dependency order."""
    path, error = _find_workflow_path(workflow_id, _session)
    if error:
        return {"summary": error, "error": "not_found"}
    payload, error = _load_workflow_payload(path)
    if error or payload is None:
        return {"summary": error or "Invalid workflow payload.", "error": "workflow_error"}

    cleaned_steps, step_error = _normalize_workflow_steps(payload.get("steps", []))
    if step_error:
        return {"summary": f"Workflow invalid: {step_error}", "error": "invalid_steps"}

    if dry_run:
        return {
            "summary": f"[DRY RUN] Workflow '{payload.get('name', path.stem)}' ready with {len(cleaned_steps)} steps.",
            "workflow_id": path.stem,
            "name": payload.get("name", path.stem),
            "steps": cleaned_steps,
            "path": str(path),
        }

    from ct.tools import registry as _tool_registry

    results = {}
    status_by_id = {step["id"]: "pending" for step in cleaned_steps}
    step_index = {step["id"]: step for step in cleaned_steps}
    loop_guard = len(cleaned_steps) * 4 + 8
    iterations = 0
    executed = []

    while iterations < loop_guard:
        iterations += 1
        progressed = False

        for step in cleaned_steps:
            sid = step["id"]
            if status_by_id[sid] != "pending":
                continue
            deps = step.get("depends_on", [])
            if any(status_by_id.get(dep) not in {"completed"} for dep in deps):
                # If a dependency failed and we're strict, abort this step.
                if any(status_by_id.get(dep) == "failed" for dep in deps) and not continue_on_error:
                    status_by_id[sid] = "skipped"
                continue

            tool_name = step["tool"]
            tool = _tool_registry.get_tool(tool_name)
            if tool is None:
                status_by_id[sid] = "failed"
                results[sid] = {"error": "tool_not_found", "summary": f"Tool not found: {tool_name}"}
                if not continue_on_error:
                    return {
                        "summary": f"Workflow failed at step {sid}: tool not found ({tool_name}).",
                        "workflow_id": path.stem,
                        "results": results,
                        "status_by_step": status_by_id,
                    }
                progressed = True
                continue

            args = dict(step.get("tool_args", {}))
            for key, val in list(args.items()):
                if isinstance(val, str) and val.startswith("$step."):
                    parts = val.split(".")
                    if len(parts) < 2:
                        continue
                    ref_id = _parse_int(parts[1], -1)
                    if ref_id not in results:
                        continue
                    resolved = results[ref_id]
                    for field in parts[2:]:
                        if isinstance(resolved, dict) and field in resolved:
                            resolved = resolved[field]
                        else:
                            break
                    args[key] = resolved

            args["_session"] = _session
            args["_prior_results"] = dict(_prior_results or {}) | results
            try:
                result = tool.run(**args)
            except Exception as exc:
                result = {"error": "execution_exception", "summary": f"{tool_name} crashed: {exc}"}

            has_error = isinstance(result, dict) and result.get("error")
            results[sid] = result
            status_by_id[sid] = "failed" if has_error else "completed"
            executed.append({"step_id": sid, "tool": tool_name, "status": status_by_id[sid]})
            progressed = True

            if has_error and not continue_on_error:
                return {
                    "summary": f"Workflow failed at step {sid} ({tool_name}).",
                    "workflow_id": path.stem,
                    "name": payload.get("name", path.stem),
                    "results": results,
                    "status_by_step": status_by_id,
                    "executed": executed,
                }

        if not progressed:
            break

    pending = [sid for sid, st in status_by_id.items() if st == "pending"]
    if pending:
        for sid in pending:
            # unresolved dependencies / cycle
            status_by_id[sid] = "skipped"

    completed = sum(1 for st in status_by_id.values() if st == "completed")
    failed = sum(1 for st in status_by_id.values() if st == "failed")
    skipped = sum(1 for st in status_by_id.values() if st == "skipped")
    summary = (
        f"Workflow '{payload.get('name', path.stem)}' executed: "
        f"{completed} completed, {failed} failed, {skipped} skipped."
    )
    return {
        "summary": summary,
        "workflow_id": path.stem,
        "name": payload.get("name", path.stem),
        "results": results,
        "status_by_step": status_by_id,
        "executed": executed,
        "path": str(path),
    }
