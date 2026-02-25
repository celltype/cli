"""
Data models for the prompt optimization module.
"""

from dataclasses import dataclass, field


@dataclass
class BudgetConfig:
    """Limits for a prompt optimization run."""
    max_iterations: int = 3
    population_size: int = 4
    elite_count: int = 2
    eval_samples: int = 1
    max_cost_usd: float = 0.50
    max_tokens: int = 500_000
    fitness_threshold: float = 0.85


RUBRIC_DIMENSIONS = [
    "specificity",
    "structure",
    "mechanistic_depth",
    "actionability",
    "completeness",
]

MAX_RUBRIC_SCORE = len(RUBRIC_DIMENSIONS) * 3  # 5 dims * 3 pts = 15


@dataclass
class RubricScore:
    """Score on a single rubric dimension (0-3)."""
    dimension: str
    score: int
    reasoning: str


@dataclass
class PromptCandidate:
    """A single prompt variant in the population."""
    text: str
    generation: int = 0
    scores: list[RubricScore] = field(default_factory=list)
    fitness: float = 0.0
    mutation_strategy: str = ""

    @property
    def fitness_normalized(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / MAX_RUBRIC_SCORE


@dataclass
class PromptOptimizationResult:
    """Final output of the optimization run."""
    original_prompt: str
    best_prompt: str
    best_fitness: float
    original_fitness: float
    best_scores: list[RubricScore]
    generations_run: int
    total_candidates_evaluated: int
    total_tokens_used: int
    total_cost_usd: float
    stop_reason: str
    early_stopped: bool
