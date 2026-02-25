"""
Budget manager for prompt optimization — enforces token, cost, and iteration caps.
"""

from ct.prompt_optimization.models import BudgetConfig


class BudgetManager:
    def __init__(self, config: BudgetConfig):
        self.config = config
        self._tokens_used: int = 0
        self._cost_usd: float = 0.0
        self._iterations_completed: int = 0
        self._candidates_evaluated: int = 0

    def sync_from_usage(self, usage_tracker) -> None:
        """Pull current totals from an LLMClient's UsageTracker."""
        self._tokens_used = usage_tracker.total_tokens
        self._cost_usd = usage_tracker.total_cost

    def record_iteration(self) -> None:
        self._iterations_completed += 1

    def record_evaluation(self) -> None:
        self._candidates_evaluated += 1

    def is_exhausted(self) -> tuple[bool, str]:
        """Check if any budget limit is exceeded. Returns (exhausted, reason)."""
        if self._iterations_completed >= self.config.max_iterations:
            return True, "max_iterations"
        if self._cost_usd >= self.config.max_cost_usd:
            return True, "budget_cost"
        if self._tokens_used >= self.config.max_tokens:
            return True, "budget_tokens"
        return False, ""

    def can_afford_generation(self) -> bool:
        """Rough heuristic: enough headroom for one more generation?"""
        estimated_tokens = self.config.population_size * 4000
        return (self._tokens_used + estimated_tokens) < self.config.max_tokens

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    @property
    def cost_usd(self) -> float:
        return self._cost_usd

    @property
    def iterations_completed(self) -> int:
        return self._iterations_completed

    @property
    def candidates_evaluated(self) -> int:
        return self._candidates_evaluated

    def summary(self) -> str:
        return (
            f"{self._iterations_completed}/{self.config.max_iterations} generations | "
            f"{self._tokens_used:,} tokens | "
            f"${self._cost_usd:.3f}/${self.config.max_cost_usd:.2f}"
        )
