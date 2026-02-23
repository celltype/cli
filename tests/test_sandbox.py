"""Tests for the sandboxed code execution environment."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def sandbox(tmp_path):
    from ct.agent.sandbox import Sandbox
    return Sandbox(timeout=5, output_dir=tmp_path)


class TestBasicExecution:
    def test_simple_math(self, sandbox):
        result = sandbox.execute("result = {'summary': 'ok', 'value': 2 + 2}")
        assert result["error"] is None
        assert result["result"]["value"] == 4
        assert result["result"]["summary"] == "ok"

    def test_stdout_capture(self, sandbox):
        result = sandbox.execute("print('hello world')")
        assert result["error"] is None
        assert "hello world" in result["stdout"]

    def test_namespace_has_libraries(self, sandbox):
        code = """
import pandas
result = {
    'summary': 'libs ok',
    'has_pd': 'pd' in dir(),
    'has_np': 'np' in dir(),
    'has_plt': 'plt' in dir(),
}
"""
        # pd, np, plt are injected directly — verify by using them
        code2 = """
df = pd.DataFrame({'a': [1, 2, 3]})
arr = np.array([1, 2, 3])
result = {'summary': 'ok', 'sum': int(df['a'].sum()), 'np_sum': int(arr.sum())}
"""
        result = sandbox.execute(code2)
        assert result["error"] is None
        assert result["result"]["sum"] == 6
        assert result["result"]["np_sum"] == 6


class TestErrors:
    def test_timeout_triggers(self, sandbox):
        result = sandbox.execute("while True: pass")
        assert result["error"] is not None
        assert "timed out" in result["error"].lower() or "timeout" in result["error"].lower()

    def test_syntax_error(self, sandbox):
        result = sandbox.execute("def foo(")
        assert result["error"] is not None
        assert "SyntaxError" in result["error"]

    def test_runtime_error(self, sandbox):
        result = sandbox.execute("x = 1 / 0")
        assert result["error"] is not None
        assert "ZeroDivisionError" in result["error"]


class TestSafety:
    def test_blocked_subprocess(self, sandbox):
        result = sandbox.execute("import subprocess")
        assert result["error"] is not None
        assert "blocked" in result["error"].lower()

    def test_blocked_socket(self, sandbox):
        result = sandbox.execute("import socket")
        assert result["error"] is not None
        assert "blocked" in result["error"].lower()

    def test_blocked_os_module(self, sandbox):
        result = sandbox.execute("import os")
        assert result["error"] is not None
        assert "blocked" in result["error"].lower()

    def test_allowed_scipy(self, sandbox):
        result = sandbox.execute("""
from scipy import stats
stat, p = stats.pearsonr([1,2,3,4], [1,2,3,4])
result = {'summary': f'r={stat:.2f}, p={p:.4f}', 'r': stat}
""")
        assert result["error"] is None
        assert result["result"]["r"] == pytest.approx(1.0)


class TestFileOutput:
    def test_matplotlib_saves_plot(self, sandbox):
        code = """
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
ax.set_title("Test")
plt.savefig(OUTPUT_DIR / "test_plot.png", dpi=72)
plt.close()
result = {'summary': 'Plot saved'}
"""
        result = sandbox.execute(code)
        assert result["error"] is None
        assert len(result["plots"]) == 1
        assert "test_plot.png" in result["plots"][0]
        assert Path(result["plots"][0]).exists()

    def test_write_outside_output_dir_blocked(self, sandbox, tmp_path):
        outside = tmp_path.parent / "sandbox_outside_write.txt"
        code = f"""
with open(r"{outside}", "w") as f:
    f.write("blocked")
result = {{"summary": "unexpected success"}}
"""
        result = sandbox.execute(code)
        assert result["error"] is not None
        assert "restricted" in result["error"].lower()
        assert not outside.exists()

    def test_write_inside_output_dir_allowed(self, sandbox):
        inside = sandbox.output_dir / "allowed_write.txt"
        code = f"""
with open(r"{inside}", "w") as f:
    f.write("ok")
result = {{"summary": "write ok"}}
"""
        result = sandbox.execute(code)
        assert result["error"] is None
        assert inside.exists()
        assert inside.read_text() == "ok"

    def test_read_outside_allowed_dirs_blocked(self, sandbox, tmp_path):
        outside = tmp_path.parent / "sandbox_outside_read.txt"
        outside.write_text("secret")
        code = f"""
with open(r"{outside}", "r") as f:
    txt = f.read()
result = {{"summary": txt}}
"""
        result = sandbox.execute(code)
        assert result["error"] is not None
        assert "restricted" in result["error"].lower()


class TestDataInjection:
    def test_inject_dataframe(self, sandbox):
        df = pd.DataFrame({"gene": ["TP53", "BRCA1"], "score": [-1.0, -0.5]})
        sandbox._namespace["test_data"] = df
        result = sandbox.execute("""
result = {'summary': f'{len(test_data)} rows', 'n': len(test_data)}
""")
        assert result["error"] is None
        assert result["result"]["n"] == 2

    def test_inject_prior_results(self, sandbox):
        sandbox.inject_prior_results({
            1: {"summary": "found 5 genes", "genes": ["TP53", "BRCA1"]},
            2: {"summary": "drug sensitivity data", "n_compounds": 10},
        })
        result = sandbox.execute("""
genes = step_1['genes']
result = {'summary': f'Got {len(genes)} genes from step 1', 'genes': genes}
""")
        assert result["error"] is None
        assert result["result"]["genes"] == ["TP53", "BRCA1"]


class TestDescribeNamespace:
    def test_describe_contains_libraries(self, sandbox):
        desc = sandbox.describe_namespace()
        assert "pd" in desc
        assert "np" in desc
        assert "plt" in desc
        assert "OUTPUT_DIR" in desc

    def test_describe_shows_datasets(self, sandbox):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        sandbox._namespace["crispr"] = df
        desc = sandbox.describe_namespace()
        assert "crispr" in desc
        assert "2 rows" in desc

    def test_describe_shows_prior_results(self, sandbox):
        sandbox.inject_prior_results({1: {"summary": "test", "data": [1, 2, 3]}})
        desc = sandbox.describe_namespace()
        assert "step_1" in desc
