"""
Domain benchmark harness and release gating.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

try:
    from ct.agent.quality import evaluate_synthesis_quality
except ImportError:
    evaluate_synthesis_quality = None


@dataclass
class BenchmarkCase:
    name: str
    domain: str
    synthesis: str
    completed_step_ids: list[int]
    expect_pass: bool = True


@dataclass
class BenchmarkResult:
    name: str
    domain: str
    passed: bool
    expected_pass: bool
    issues: list[str]


class BenchmarkSuite:
    """Runs deterministic benchmark cases for synthesis quality gates."""

    def __init__(self, cases: list[BenchmarkCase]):
        self.cases = cases

    @classmethod
    def load(cls, path: Path | None = None) -> "BenchmarkSuite":
        source = path or (Path.cwd() / "configs" / "pharma_benchmarks.json")
        if source.exists():
            data = json.loads(source.read_text(encoding="utf-8"))
            raw_cases = data.get("cases", [])
            cases = [BenchmarkCase(**item) for item in raw_cases]
            if cases:
                return cls(cases)
        return cls(default_cases())

    def run(self) -> dict[str, Any]:
        results: list[BenchmarkResult] = []
        for case in self.cases:
            quality = evaluate_synthesis_quality(
                case.synthesis,
                completed_step_ids=set(case.completed_step_ids),
                require_key_evidence=True,
                min_next_steps=2,
                max_next_steps=3,
            )
            passed = quality.ok
            results.append(
                BenchmarkResult(
                    name=case.name,
                    domain=case.domain,
                    passed=passed,
                    expected_pass=case.expect_pass,
                    issues=quality.issues,
                )
            )

        expected_correct = 0
        for result in results:
            if result.passed == result.expected_pass:
                expected_correct += 1
        total = len(results)
        pass_rate = expected_correct / max(total, 1)
        domain_scores = self._domain_scores(results)
        return {
            "total_cases": total,
            "expected_behavior_matches": expected_correct,
            "pass_rate": round(pass_rate, 4),
            "domain_scores": domain_scores,
            "results": [asdict(r) for r in results],
        }

    @staticmethod
    def _domain_scores(results: list[BenchmarkResult]) -> dict[str, float]:
        buckets: dict[str, list[bool]] = {}
        for result in results:
            buckets.setdefault(result.domain, []).append(result.passed == result.expected_pass)
        scores = {}
        for domain, vals in buckets.items():
            scores[domain] = round(sum(1 for v in vals if v) / max(len(vals), 1), 4)
        return scores

    @staticmethod
    def gate(summary: dict[str, Any], *, min_pass_rate: float = 0.9) -> dict[str, Any]:
        actual = float(summary.get("pass_rate", 0.0))
        ok = actual >= min_pass_rate
        return {
            "ok": ok,
            "min_pass_rate": min_pass_rate,
            "actual_pass_rate": actual,
            "message": (
                f"Benchmark gate passed ({actual:.2%} >= {min_pass_rate:.2%})"
                if ok
                else f"Benchmark gate failed ({actual:.2%} < {min_pass_rate:.2%})"
            ),
        }


def default_cases() -> list[BenchmarkCase]:
    """Fallback deterministic cases when benchmark file is absent."""
    return [
        BenchmarkCase(
            name="target_validation_grounded",
            domain="target_validation",
            synthesis=(
                "## Answer\nSignal supports target progression.\n\n"
                "## Key Evidence\n- Genetic support observed [step:1]\n"
                "- Expression concordance supports mechanism [step:2]\n\n"
                "## Confidence & Caveats\n- Moderate confidence.\n\n"
                "## Suggested Next Steps\n"
                "1. Run genomics.coloc for the top locus.\n"
                "2. Run target.disease_association for indication prioritization.\n"
            ),
            completed_step_ids=[1, 2],
            expect_pass=True,
        ),
        BenchmarkCase(
            name="moa_missing_citation",
            domain="moa_inference",
            synthesis=(
                "## Answer\nMechanism inferred.\n\n"
                "## Key Evidence\n- Strong MOA pattern without citation\n\n"
                "## Confidence & Caveats\n- Preliminary.\n\n"
                "## Suggested Next Steps\n"
                "1. Run expression.pathway_enrichment for the top signature.\n"
                "2. Run repurposing.cmap_query on the differential profile.\n"
            ),
            completed_step_ids=[1, 2],
            expect_pass=False,
        ),
    ]
