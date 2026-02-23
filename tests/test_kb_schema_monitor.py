"""Tests for schema drift monitor."""

from ct.kb.schema_monitor import SchemaMonitor


def test_schema_monitor_new_then_ok_then_drift(tmp_path):
    state = {"v": 1}

    def probe():
        return {"a": 1, "b": {"x": "y"}, "list": [state["v"]]}

    monitor = SchemaMonitor(
        baseline_path=tmp_path / "baseline.json",
        monitors={"probe": probe},
    )

    initial = monitor.check()
    assert initial[0].status == "new"

    monitor.update_baseline()
    second = monitor.check()
    assert second[0].status == "ok"

    state["v"] = {"nested": True}
    third = monitor.check()
    assert third[0].status == "drift"
    assert third[0].added_paths or third[0].removed_paths
