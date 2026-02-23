"""
Schema drift monitor for external API/data integrations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable


@dataclass
class SchemaCheckResult:
    monitor: str
    status: str  # ok | drift | new | error
    added_paths: list[str]
    removed_paths: list[str]
    baseline_size: int
    current_size: int
    error: str = ""


def _schema_paths(obj: Any, prefix: str = "$") -> set[str]:
    """Flatten JSON-like object into typed path signatures."""
    paths = set()
    if isinstance(obj, dict):
        paths.add(f"{prefix}:object")
        for key, value in obj.items():
            child = f"{prefix}.{key}"
            paths.update(_schema_paths(value, child))
        return paths
    if isinstance(obj, list):
        paths.add(f"{prefix}:array")
        if obj:
            # Sample first few elements for schema signature.
            for item in obj[:3]:
                paths.update(_schema_paths(item, f"{prefix}[]"))
        return paths
    if obj is None:
        paths.add(f"{prefix}:null")
        return paths
    typename = type(obj).__name__
    paths.add(f"{prefix}:{typename}")
    return paths


class SchemaMonitor:
    """Capture and compare tool output schemas against baselines."""

    def __init__(
        self,
        baseline_path: Path | None = None,
        monitors: dict[str, Callable[[], Any]] | None = None,
    ):
        self.baseline_path = baseline_path or (Path.home() / ".ct" / "knowledge" / "schema_baselines.json")
        self.monitors = monitors or self._default_monitors()
        self._baseline = self._load_baseline()

    def _default_monitors(self) -> dict[str, Callable[[], Any]]:
        from ct.tools.data_api import opentargets_search, uniprot_lookup
        from ct.tools.literature import openalex_search, pubmed_search

        return {
            "literature.pubmed_search": lambda: pubmed_search("TP53 cancer", max_results=1),
            "literature.openalex_search": lambda: openalex_search("TP53 cancer", max_results=1),
            "data_api.uniprot_lookup": lambda: uniprot_lookup("P04637"),
            "data_api.opentargets_search": lambda: opentargets_search("TP53", entity_type="target"),
        }

    def _load_baseline(self) -> dict[str, Any]:
        if not self.baseline_path.exists():
            return {"version": 1, "monitors": {}}
        try:
            data = json.loads(self.baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "monitors": {}}
        if not isinstance(data, dict):
            return {"version": 1, "monitors": {}}
        data.setdefault("version", 1)
        data.setdefault("monitors", {})
        return data

    def save_baseline(self):
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self.baseline_path.write_text(
            json.dumps(self._baseline, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def update_baseline(self, *, monitor: str | None = None) -> list[SchemaCheckResult]:
        """Capture current schema(s) as new baseline."""
        results = self.check(update_baseline=True, monitor=monitor)
        self.save_baseline()
        return results

    def check(
        self,
        *,
        update_baseline: bool = False,
        monitor: str | None = None,
    ) -> list[SchemaCheckResult]:
        """Run schema checks and return diff results."""
        results = []
        for name, probe in self.monitors.items():
            if monitor and name != monitor:
                continue

            baseline_paths = set(self._baseline["monitors"].get(name, {}).get("paths", []))
            try:
                payload = probe()
            except Exception as exc:
                results.append(
                    SchemaCheckResult(
                        monitor=name,
                        status="error",
                        added_paths=[],
                        removed_paths=[],
                        baseline_size=len(baseline_paths),
                        current_size=0,
                        error=str(exc),
                    )
                )
                continue

            current_paths = _schema_paths(payload)
            added = sorted(current_paths - baseline_paths)
            removed = sorted(baseline_paths - current_paths)
            if not baseline_paths:
                status = "new"
            elif not added and not removed:
                status = "ok"
            else:
                status = "drift"

            results.append(
                SchemaCheckResult(
                    monitor=name,
                    status=status,
                    added_paths=added,
                    removed_paths=removed,
                    baseline_size=len(baseline_paths),
                    current_size=len(current_paths),
                )
            )

            if update_baseline:
                self._baseline["monitors"][name] = {
                    "paths": sorted(current_paths),
                }
        return results

    @staticmethod
    def summarize(results: list[SchemaCheckResult]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for result in results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return {
            "total": len(results),
            "counts": counts,
            "results": [asdict(r) for r in results],
        }
