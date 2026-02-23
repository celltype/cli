"""Tests for benchmark suite and release gate."""

from ct.kb.benchmarks import BenchmarkCase, BenchmarkSuite


def test_benchmark_suite_run_and_gate():
    suite = BenchmarkSuite(
        [
            BenchmarkCase(
                name="good",
                domain="target_validation",
                synthesis=(
                    "## Answer\nA\n\n## Key Evidence\n- E [step:1]\n\n"
                    "## Confidence & Caveats\n- C\n\n"
                    "## Suggested Next Steps\n"
                    "1. Run genomics.coloc for this target.\n"
                    "2. Run target.disease_association for top diseases.\n"
                ),
                completed_step_ids=[1],
                expect_pass=True,
            ),
            BenchmarkCase(
                name="bad",
                domain="target_validation",
                synthesis=(
                    "## Answer\nA\n\n## Key Evidence\n- no cite\n\n"
                    "## Confidence & Caveats\n- C\n\n"
                    "## Suggested Next Steps\n"
                    "1. Do more analysis.\n"
                    "2. Consider next steps.\n"
                ),
                completed_step_ids=[1],
                expect_pass=False,
            ),
        ]
    )
    summary = suite.run()
    assert summary["total_cases"] == 2
    gate = suite.gate(summary, min_pass_rate=0.5)
    assert gate["ok"] is True
