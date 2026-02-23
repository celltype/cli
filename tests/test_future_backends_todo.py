"""Guardrail tests for future model backend integrations.

These backends are tracked in docs/model_backends_todo.md and must not appear
as registered tools until implementation/runtime gates are complete.
"""

from ct.tools import ensure_loaded, registry


def test_future_backend_tools_not_registered_yet():
    ensure_loaded()
    planned_tools = [
        "structure.boltz_predict",
        "structure.chai_predict",
        "protein.mpnn_design",
        "protein.ligand_mpnn_design",
        "protein.thermo_mpnn_optimize",
    ]

    for name in planned_tools:
        assert registry.get_tool(name) is None, f"Future backend tool unexpectedly registered: {name}"
