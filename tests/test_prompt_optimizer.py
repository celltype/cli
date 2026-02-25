"""
Tests for prompt optimizer: models, budget, mutator, evaluator, and optimizer loop.

All tests mock LLMClient.chat() — no real API calls.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from ct.prompt_optimization.models import (
    BudgetConfig,
    PromptCandidate,
    RubricScore,
    PromptOptimizationResult,
    MAX_RUBRIC_SCORE,
    RUBRIC_DIMENSIONS,
)
from ct.prompt_optimization.budget import BudgetManager
from ct.prompt_optimization.mutator import Mutator
from ct.prompt_optimization.evaluator import Evaluator
from ct.prompt_optimization.optimizer import PromptOptimizer
from ct.models.llm import LLMClient, LLMResponse, UsageTracker


# ── Helpers ──────────────────────────────────────────────────

def _make_eval_json(scores: dict[str, int] | None = None) -> str:
    """Build evaluator JSON response string."""
    scores = scores or {d: 2 for d in RUBRIC_DIMENSIONS}
    return json.dumps({
        "scores": [
            {"dimension": d, "score": s, "reasoning": f"{d} is ok"}
            for d, s in scores.items()
        ]
    })


def _make_eval_response(scores: dict[str, int] | None = None) -> LLMResponse:
    return LLMResponse(
        content=_make_eval_json(scores),
        model="test",
        usage={"input": 200, "output": 100},
    )


def _make_seed_response(count: int = 3) -> LLMResponse:
    variants = [f"variant {i}" for i in range(1, count + 1)]
    return LLMResponse(
        content=json.dumps(variants),
        model="test",
        usage={"input": 100, "output": 50},
    )


def _make_mutate_response(text: str = "mutated prompt") -> LLMResponse:
    return LLMResponse(
        content=text,
        model="test",
        usage={"input": 100, "output": 50},
    )


# ── Model tests ─────────────────────────────────────────────

class TestPromptCandidate:
    def test_fitness_normalized_empty(self):
        c = PromptCandidate(text="test")
        assert c.fitness_normalized == 0.0

    def test_fitness_normalized_perfect(self):
        c = PromptCandidate(
            text="test",
            scores=[RubricScore(d, 3, "") for d in RUBRIC_DIMENSIONS],
        )
        assert c.fitness_normalized == 1.0

    def test_fitness_normalized_partial(self):
        scores = [RubricScore(d, 1, "") for d in RUBRIC_DIMENSIONS]
        c = PromptCandidate(text="test", scores=scores)
        assert c.fitness_normalized == pytest.approx(5 / MAX_RUBRIC_SCORE)

    def test_fitness_normalized_mixed(self):
        scores = [
            RubricScore("specificity", 3, ""),
            RubricScore("structure", 0, ""),
            RubricScore("mechanistic_depth", 2, ""),
            RubricScore("actionability", 1, ""),
            RubricScore("completeness", 3, ""),
        ]
        c = PromptCandidate(text="test", scores=scores)
        assert c.fitness_normalized == pytest.approx(9 / 15)


class TestBudgetConfig:
    def test_defaults(self):
        bc = BudgetConfig()
        assert bc.max_iterations == 3
        assert bc.population_size == 4
        assert bc.elite_count == 2
        assert bc.max_cost_usd == 0.50


# ── Budget tests ─────────────────────────────────────────────

class TestBudgetManager:
    def test_fresh_not_exhausted(self):
        bm = BudgetManager(BudgetConfig(max_iterations=3))
        exhausted, reason = bm.is_exhausted()
        assert not exhausted
        assert reason == ""

    def test_iteration_limit(self):
        bm = BudgetManager(BudgetConfig(max_iterations=2))
        bm.record_iteration()
        bm.record_iteration()
        exhausted, reason = bm.is_exhausted()
        assert exhausted
        assert reason == "max_iterations"

    def test_cost_limit(self):
        bm = BudgetManager(BudgetConfig(max_cost_usd=0.10))
        bm._cost_usd = 0.15
        exhausted, reason = bm.is_exhausted()
        assert exhausted
        assert reason == "budget_cost"

    def test_token_limit(self):
        bm = BudgetManager(BudgetConfig(max_tokens=1000))
        bm._tokens_used = 1500
        exhausted, reason = bm.is_exhausted()
        assert exhausted
        assert reason == "budget_tokens"

    def test_summary_format(self):
        bm = BudgetManager(BudgetConfig())
        s = bm.summary()
        assert "0/3 generations" in s
        assert "$" in s

    def test_can_afford_generation(self):
        bm = BudgetManager(BudgetConfig(max_tokens=100_000, population_size=4))
        assert bm.can_afford_generation()
        bm._tokens_used = 99_000
        assert not bm.can_afford_generation()

    def test_record_evaluation(self):
        bm = BudgetManager(BudgetConfig())
        assert bm.candidates_evaluated == 0
        bm.record_evaluation()
        bm.record_evaluation()
        assert bm.candidates_evaluated == 2


# ── Mutator tests ────────────────────────────────────────────

class TestMutator:
    def test_seed_variants_parses_json(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat.return_value = _make_seed_response(3)
        m = Mutator(mock_llm)
        variants = m.seed_variants("test prompt", count=3)
        assert len(variants) == 3
        assert variants[0] == "variant 1"
        assert variants[2] == "variant 3"

    def test_seed_variants_with_markdown_fences(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat.return_value = LLMResponse(
            content='```json\n["v1", "v2"]\n```',
            model="test",
            usage={"input": 100, "output": 50},
        )
        m = Mutator(mock_llm)
        variants = m.seed_variants("test", count=2)
        assert len(variants) == 2
        assert variants[0] == "v1"

    def test_seed_variants_json_fallback(self):
        """When JSON parsing fails, falls back to individual mutations."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat.side_effect = [
            LLMResponse(content="not json at all", model="test", usage={"input": 100, "output": 50}),
            _make_mutate_response("fallback 1"),
            _make_mutate_response("fallback 2"),
        ]
        m = Mutator(mock_llm)
        variants = m.seed_variants("test", count=2)
        assert len(variants) == 2
        assert variants[0] == "fallback 1"

    def test_mutate_returns_text(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat.return_value = _make_mutate_response("improved prompt")
        m = Mutator(mock_llm)
        assert m.mutate("original") == "improved prompt"

    def test_crossover_returns_text(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat.return_value = _make_mutate_response("crossed prompt")
        m = Mutator(mock_llm)
        assert m.crossover("a", "b") == "crossed prompt"


# ── Evaluator tests ──────────────────────────────────────────

class TestEvaluator:
    def test_evaluate_parses_scores(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat.return_value = _make_eval_response()
        e = Evaluator(mock_llm)
        c = PromptCandidate(text="test")
        result = e.evaluate(c, num_samples=1)
        assert len(result.scores) == 5
        assert result.fitness == pytest.approx(10 / 15)

    def test_evaluate_with_markdown_fences(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat.return_value = LLMResponse(
            content=f"```json\n{_make_eval_json()}\n```",
            model="test",
            usage={"input": 200, "output": 100},
        )
        e = Evaluator(mock_llm)
        c = PromptCandidate(text="test")
        result = e.evaluate(c, num_samples=1)
        assert len(result.scores) == 5

    def test_evaluate_json_fallback(self):
        """Malformed JSON returns default scores."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat.return_value = LLMResponse(
            content="not valid json",
            model="test",
            usage={"input": 200, "output": 100},
        )
        e = Evaluator(mock_llm)
        c = PromptCandidate(text="test")
        result = e.evaluate(c, num_samples=1)
        assert len(result.scores) == 5
        assert all(s.score == 1 for s in result.scores)

    def test_self_consistency_averaging(self):
        mock_llm = MagicMock(spec=LLMClient)
        resp1 = _make_eval_response({d: 2 for d in RUBRIC_DIMENSIONS})
        resp2 = _make_eval_response({d: 3 for d in RUBRIC_DIMENSIONS})
        mock_llm.chat.side_effect = [resp1, resp2]
        e = Evaluator(mock_llm)
        c = PromptCandidate(text="test")
        result = e.evaluate(c, num_samples=2)
        # avg of 2 and 3 → rounds to 2 or 3
        assert all(s.score in (2, 3) for s in result.scores)

    def test_evaluate_clamps_scores(self):
        """Scores outside 0-3 are clamped."""
        mock_llm = MagicMock(spec=LLMClient)
        bad_json = json.dumps({"scores": [
            {"dimension": d, "score": 5 if i == 0 else -1, "reasoning": "clamped"}
            for i, d in enumerate(RUBRIC_DIMENSIONS)
        ]})
        mock_llm.chat.return_value = LLMResponse(
            content=bad_json, model="test", usage={"input": 200, "output": 100},
        )
        e = Evaluator(mock_llm)
        c = PromptCandidate(text="test")
        result = e.evaluate(c, num_samples=1)
        assert result.scores[0].score == 3  # clamped from 5
        assert all(s.score >= 0 for s in result.scores)

    def test_evaluate_missing_dimensions_filled(self):
        """Missing dimensions get default score of 1."""
        mock_llm = MagicMock(spec=LLMClient)
        partial_json = json.dumps({"scores": [
            {"dimension": "specificity", "score": 3, "reasoning": "good"},
            {"dimension": "structure", "score": 2, "reasoning": "ok"},
        ]})
        mock_llm.chat.return_value = LLMResponse(
            content=partial_json, model="test", usage={"input": 200, "output": 100},
        )
        e = Evaluator(mock_llm)
        c = PromptCandidate(text="test")
        result = e.evaluate(c, num_samples=1)
        assert len(result.scores) == 5

    def test_evaluate_llm_error_returns_defaults(self):
        """LLM errors produce default scores instead of crashing."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.chat.side_effect = Exception("API error")
        e = Evaluator(mock_llm)
        c = PromptCandidate(text="test")
        result = e.evaluate(c, num_samples=1)
        assert len(result.scores) == 5
        assert all(s.score == 1 for s in result.scores)


# ── Optimizer integration tests ──────────────────────────────

class TestPromptOptimizer:
    def _make_optimizer(self, mock_llm, **budget_kwargs):
        from rich.console import Console
        import io
        quiet_console = Console(file=io.StringIO())  # suppress output
        budget = BudgetConfig(**budget_kwargs)
        return PromptOptimizer(llm=mock_llm, budget_config=budget, console=quiet_console)

    def test_single_iteration_returns_result(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.usage = UsageTracker()

        mock_llm.chat.side_effect = [
            _make_seed_response(3),    # seed variants
        ] + [_make_eval_response()] * 20  # evaluations

        optimizer = self._make_optimizer(mock_llm, max_iterations=1, population_size=4)
        result = optimizer.optimize("test drug discovery prompt")

        assert isinstance(result, PromptOptimizationResult)
        assert result.original_prompt == "test drug discovery prompt"
        assert result.best_prompt
        assert 0 <= result.best_fitness <= 1.0
        assert result.generations_run >= 1

    def test_original_prompt_always_present(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.usage = UsageTracker()

        mock_llm.chat.side_effect = [
            _make_seed_response(3),
        ] + [_make_eval_response()] * 20

        optimizer = self._make_optimizer(mock_llm, max_iterations=1, population_size=4)
        result = optimizer.optimize("my original prompt")

        assert result.original_prompt == "my original prompt"
        assert result.original_fitness >= 0

    def test_early_stop_on_high_fitness(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.usage = UsageTracker()

        perfect_scores = {d: 3 for d in RUBRIC_DIMENSIONS}
        mock_llm.chat.side_effect = [
            _make_seed_response(3),
        ] + [_make_eval_response(perfect_scores)] * 20

        optimizer = self._make_optimizer(
            mock_llm, max_iterations=5, population_size=4, fitness_threshold=0.85
        )
        result = optimizer.optimize("test")

        assert result.early_stopped
        assert result.stop_reason == "fitness_threshold"
        assert result.generations_run == 1  # should stop after gen 0

    def test_returns_best_candidate(self):
        """Best candidate has highest fitness."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.usage = UsageTracker()

        low_scores = {d: 1 for d in RUBRIC_DIMENSIONS}
        high_scores = {d: 3 for d in RUBRIC_DIMENSIONS}

        mock_llm.chat.side_effect = [
            _make_seed_response(3),
            _make_eval_response(low_scores),   # original
            _make_eval_response(high_scores),  # variant 1 (best)
            _make_eval_response(low_scores),   # variant 2
            _make_eval_response(low_scores),   # variant 3
        ] + [_make_eval_response(low_scores)] * 20  # further generations

        optimizer = self._make_optimizer(
            mock_llm, max_iterations=1, population_size=4, fitness_threshold=1.0,
        )
        result = optimizer.optimize("test")

        assert result.best_fitness == pytest.approx(1.0)

    def test_result_has_all_fields(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.usage = UsageTracker()

        mock_llm.chat.side_effect = [
            _make_seed_response(1),
        ] + [_make_eval_response()] * 10

        optimizer = self._make_optimizer(
            mock_llm, max_iterations=1, population_size=2
        )
        result = optimizer.optimize("test")

        assert hasattr(result, "original_prompt")
        assert hasattr(result, "best_prompt")
        assert hasattr(result, "best_fitness")
        assert hasattr(result, "original_fitness")
        assert hasattr(result, "best_scores")
        assert hasattr(result, "generations_run")
        assert hasattr(result, "total_candidates_evaluated")
        assert hasattr(result, "total_tokens_used")
        assert hasattr(result, "total_cost_usd")
        assert hasattr(result, "stop_reason")
        assert hasattr(result, "early_stopped")
