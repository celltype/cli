"""
PromptOptimizer — evolutionary loop that improves drug discovery prompts.
"""

import logging
import random

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ct.models.llm import LLMClient
from ct.prompt_optimization.models import (
    BudgetConfig,
    PromptCandidate,
    PromptOptimizationResult,
)
from ct.prompt_optimization.budget import BudgetManager
from ct.prompt_optimization.mutator import Mutator
from ct.prompt_optimization.evaluator import Evaluator
from ct.ui.status import ThinkingStatus

logger = logging.getLogger("ct.prompt.optimizer")


class PromptOptimizer:
    def __init__(
        self,
        llm: LLMClient,
        budget_config: BudgetConfig | None = None,
        console: Console | None = None,
    ):
        self.llm = llm
        self.budget_config = budget_config or BudgetConfig()
        self.console = console or Console()
        self.budget = BudgetManager(self.budget_config)
        self.mutator = Mutator(llm)
        self.evaluator = Evaluator(llm)

    def optimize(self, raw_prompt: str) -> PromptOptimizationResult:
        """Run the full evolutionary optimization loop."""
        all_candidates: list[PromptCandidate] = []
        original_fitness = 0.0
        best = PromptCandidate(text=raw_prompt, mutation_strategy="original")
        stop_reason = "max_iterations"

        try:
            # ── Generation 0: seed ───────────────────────────
            self.console.print(
                "\n[bold cyan]Optimizing prompt...[/bold cyan] "
                f"(max {self.budget_config.max_iterations} generations, "
                f"pop {self.budget_config.population_size}, "
                f"budget ${self.budget_config.max_cost_usd:.2f})\n"
            )

            with ThinkingStatus(self.console, phase="optimizing"):
                population = self._seed_population(raw_prompt)
                population = self._evaluate_population(population)
            all_candidates.extend(population)

            # Track original prompt's fitness
            for c in population:
                if c.mutation_strategy == "original":
                    original_fitness = c.fitness
                    break

            elites = self._select_elites(population)
            best = elites[0]
            self._print_generation(0, population)
            self.budget.record_iteration()

            # Check early stop after gen 0
            if best.fitness >= self.budget_config.fitness_threshold:
                stop_reason = "fitness_threshold"
                self.console.print(
                    f"[green]Fitness threshold reached ({best.fitness:.2f} >= "
                    f"{self.budget_config.fitness_threshold})[/green]"
                )
            else:
                # ── Generations 1..N ─────────────────────────
                for gen in range(1, self.budget_config.max_iterations):
                    exhausted, reason = self.budget.is_exhausted()
                    if exhausted:
                        stop_reason = reason
                        break

                    if not self.budget.can_afford_generation():
                        stop_reason = "budget_tokens"
                        break

                    with ThinkingStatus(self.console, phase="optimizing"):
                        new_candidates = self._evolve(elites, gen)
                        new_candidates = self._evaluate_population(new_candidates)

                    # Merge elites + new, re-select
                    merged = list(elites) + new_candidates
                    all_candidates.extend(new_candidates)
                    elites = self._select_elites(merged)
                    best = elites[0]

                    self._print_generation(gen, merged)
                    self.budget.record_iteration()

                    if best.fitness >= self.budget_config.fitness_threshold:
                        stop_reason = "fitness_threshold"
                        self.console.print(
                            f"[green]Fitness threshold reached ({best.fitness:.2f})[/green]"
                        )
                        break

        except KeyboardInterrupt:
            stop_reason = "user_cancelled"
            self.console.print("\n[yellow]Optimization cancelled.[/yellow]")

        self.budget.sync_from_usage(self.llm.usage)

        result = PromptOptimizationResult(
            original_prompt=raw_prompt,
            best_prompt=best.text,
            best_fitness=best.fitness,
            original_fitness=original_fitness,
            best_scores=best.scores,
            generations_run=self.budget.iterations_completed,
            total_candidates_evaluated=self.budget.candidates_evaluated,
            total_tokens_used=self.budget.tokens_used,
            total_cost_usd=self.budget.cost_usd,
            stop_reason=stop_reason,
            early_stopped=stop_reason in ("fitness_threshold", "user_cancelled"),
        )

        self._print_result(result)
        return result

    # ── Population management ────────────────────────────────

    def _seed_population(self, raw_prompt: str) -> list[PromptCandidate]:
        """Create generation-0 population: original + LLM variants."""
        original = PromptCandidate(text=raw_prompt, generation=0, mutation_strategy="original")
        variants_needed = self.budget_config.population_size - 1

        if variants_needed <= 0:
            return [original]

        variant_texts = self.mutator.seed_variants(raw_prompt, variants_needed)
        self.budget.sync_from_usage(self.llm.usage)

        candidates = [original]
        seen = {raw_prompt.strip()}
        for text in variant_texts:
            text = text.strip()
            if text and text not in seen:
                candidates.append(
                    PromptCandidate(text=text, generation=0, mutation_strategy="seed")
                )
                seen.add(text)

        return candidates

    def _evolve(self, elites: list[PromptCandidate], generation: int) -> list[PromptCandidate]:
        """Generate new candidates from elites via mutation and crossover."""
        new_count = self.budget_config.population_size - len(elites)
        new_candidates = []
        seen = {e.text.strip() for e in elites}

        for _ in range(new_count):
            exhausted, _ = self.budget.is_exhausted()
            if exhausted:
                break

            try:
                if len(elites) >= 2 and random.random() < 0.5:
                    a, b = random.sample(elites, 2)
                    text = self.mutator.crossover(a.text, b.text)
                    strategy = "crossover"
                else:
                    parent = random.choice(elites)
                    text = self.mutator.mutate(parent.text)
                    strategy = "mutate"

                self.budget.sync_from_usage(self.llm.usage)
                text = text.strip()

                if text and text not in seen:
                    new_candidates.append(
                        PromptCandidate(text=text, generation=generation, mutation_strategy=strategy)
                    )
                    seen.add(text)
            except Exception as exc:
                logger.warning("Evolution step failed: %s", exc)

        return new_candidates

    def _evaluate_population(self, candidates: list[PromptCandidate]) -> list[PromptCandidate]:
        """Score all unevaluated candidates."""
        for c in candidates:
            if c.fitness > 0:
                continue  # already scored (elite carried forward)
            try:
                self.evaluator.evaluate(c, num_samples=self.budget_config.eval_samples)
                self.budget.sync_from_usage(self.llm.usage)
                self.budget.record_evaluation()
            except Exception as exc:
                logger.warning("Evaluation failed for candidate: %s", exc)
                c.fitness = 0.0
        return candidates

    def _select_elites(self, candidates: list[PromptCandidate]) -> list[PromptCandidate]:
        """Return top-N candidates by fitness."""
        ranked = sorted(candidates, key=lambda c: c.fitness, reverse=True)
        return ranked[: self.budget_config.elite_count]

    # ── Display ──────────────────────────────────────────────

    def _print_generation(self, gen: int, candidates: list[PromptCandidate]) -> None:
        self.budget.sync_from_usage(self.llm.usage)

        table = Table(
            title=f"Generation {gen + 1}/{self.budget_config.max_iterations}  |  {self.budget.summary()}",
            show_lines=False,
            pad_edge=False,
        )
        table.add_column("#", width=3)
        table.add_column("Fitness", width=7)
        table.add_column("Spec", width=4)
        table.add_column("Struct", width=6)
        table.add_column("Mech", width=4)
        table.add_column("Action", width=6)
        table.add_column("Compl", width=5)
        table.add_column("Strategy", width=10)

        ranked = sorted(candidates, key=lambda c: c.fitness, reverse=True)
        for i, c in enumerate(ranked, 1):
            scores_by_dim = {s.dimension: s.score for s in c.scores}
            table.add_row(
                str(i),
                f"{c.fitness:.2f}",
                str(scores_by_dim.get("specificity", "-")),
                str(scores_by_dim.get("structure", "-")),
                str(scores_by_dim.get("mechanistic_depth", "-")),
                str(scores_by_dim.get("actionability", "-")),
                str(scores_by_dim.get("completeness", "-")),
                c.mutation_strategy,
            )

        self.console.print(table)
        self.console.print()

    def _print_result(self, result: PromptOptimizationResult) -> None:
        delta = result.best_fitness - result.original_fitness
        sign = "+" if delta >= 0 else ""

        # Rubric breakdown
        rubric_lines = []
        for s in result.best_scores:
            label = s.dimension.replace("_", " ").ljust(20)
            rubric_lines.append(f"  {label} {s.score}/3  {s.reasoning}")
        rubric_text = "\n".join(rubric_lines)

        panel_text = (
            f"[bold]Original prompt:[/bold]\n{result.original_prompt}\n\n"
            f"[bold green]Optimized prompt:[/bold green]\n{result.best_prompt}\n\n"
            f"[bold]Rubric scores:[/bold]\n{rubric_text}\n\n"
            f"[dim]{result.generations_run} generations | "
            f"{result.total_candidates_evaluated} candidates | "
            f"{result.total_tokens_used:,} tokens | "
            f"${result.total_cost_usd:.3f}[/dim]"
        )

        self.console.print(
            Panel(
                panel_text,
                title=f"Prompt Optimization ({result.best_fitness:.2f} vs original {result.original_fitness:.2f}, {sign}{delta:.2f})",
                border_style="green" if delta > 0 else "yellow",
            )
        )
