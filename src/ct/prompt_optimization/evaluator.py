"""
LLM-as-judge evaluator with drug-discovery-specific rubric.
"""

import json
import logging

from ct.models.llm import LLMClient
from ct.prompt_optimization.models import PromptCandidate, RubricScore, RUBRIC_DIMENSIONS
from ct.prompt_optimization.prompts import EVAL_SYSTEM, EVAL_USER_TEMPLATE

logger = logging.getLogger("ct.prompt.evaluator")

# Default scores used when parsing fails
_DEFAULT_SCORES = [
    RubricScore(dim, 1, "evaluation parse error") for dim in RUBRIC_DIMENSIONS
]


class Evaluator:
    def __init__(self, llm: LLMClient, temperature: float = 0.3, max_tokens: int = 512):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

    def evaluate(self, candidate: PromptCandidate, num_samples: int = 1) -> PromptCandidate:
        """Score a candidate, optionally averaging over multiple eval passes."""
        all_passes: list[list[RubricScore]] = []
        for _ in range(num_samples):
            scores = self._single_eval(candidate.text)
            all_passes.append(scores)

        if num_samples == 1:
            candidate.scores = all_passes[0]
        else:
            candidate.scores = self._average_passes(all_passes)

        candidate.fitness = candidate.fitness_normalized
        return candidate

    # ── internals ────────────────────────────────────────────

    def _single_eval(self, prompt_text: str) -> list[RubricScore]:
        """One evaluation pass → list of RubricScore."""
        user_msg = EVAL_USER_TEMPLATE.format(prompt=prompt_text)
        try:
            resp = self.llm.chat(
                system=EVAL_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return self._parse_scores(resp.content)
        except Exception as exc:
            logger.warning("Evaluator LLM call failed: %s", exc)
            return list(_DEFAULT_SCORES)

    def _parse_scores(self, text: str) -> list[RubricScore]:
        """Parse evaluator JSON response into RubricScore list."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(text)
            raw_scores = data.get("scores", data) if isinstance(data, dict) else data
            if not isinstance(raw_scores, list):
                raise ValueError("Expected list of score objects")

            results = []
            for item in raw_scores:
                dim = item.get("dimension", "unknown")
                score = max(0, min(3, int(item.get("score", 1))))
                reasoning = str(item.get("reasoning", ""))
                results.append(RubricScore(dim, score, reasoning))

            # Ensure all dimensions are present
            present = {s.dimension for s in results}
            for dim in RUBRIC_DIMENSIONS:
                if dim not in present:
                    results.append(RubricScore(dim, 1, "dimension missing from eval"))

            return results[:5]  # keep exactly 5
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse evaluator JSON: %s", exc)
            return list(_DEFAULT_SCORES)

    def _average_passes(self, all_passes: list[list[RubricScore]]) -> list[RubricScore]:
        """Average scores across multiple evaluation passes per dimension."""
        dim_scores: dict[str, list[int]] = {d: [] for d in RUBRIC_DIMENSIONS}
        dim_reasons: dict[str, str] = {}

        for pass_scores in all_passes:
            for s in pass_scores:
                if s.dimension in dim_scores:
                    dim_scores[s.dimension].append(s.score)
                    dim_reasons[s.dimension] = s.reasoning  # keep last

        results = []
        for dim in RUBRIC_DIMENSIONS:
            vals = dim_scores[dim]
            avg = round(sum(vals) / len(vals)) if vals else 1
            results.append(RubricScore(dim, avg, dim_reasons.get(dim, "")))
        return results
