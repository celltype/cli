"""
LLM-driven prompt variant generation (mutation and crossover).
"""

import json
import logging

from ct.models.llm import LLMClient
from ct.prompt_optimization.prompts import (
    MUTATION_SYSTEM,
    SEED_USER_TEMPLATE,
    MUTATE_USER_TEMPLATE,
    CROSSOVER_USER_TEMPLATE,
)

logger = logging.getLogger("ct.prompt.mutator")


class Mutator:
    def __init__(self, llm: LLMClient, temperature: float = 0.9, max_tokens: int = 1024):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

    def seed_variants(self, raw_prompt: str, count: int) -> list[str]:
        """Generate *count* initial variants of the raw prompt via a single LLM call."""
        extra = (
            "4. More actionable — request specific experimental follow-ups with named assays\n"
            if count >= 4
            else ""
        )
        user_msg = SEED_USER_TEMPLATE.format(
            count=count, prompt=raw_prompt, extra_angle=extra,
        )
        resp = self.llm.chat(
            system=MUTATION_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return self._parse_variants(resp.content, raw_prompt, count)

    def mutate(self, prompt: str) -> str:
        """Apply a single mutation to a prompt."""
        user_msg = MUTATE_USER_TEMPLATE.format(prompt=prompt)
        resp = self.llm.chat(
            system=MUTATION_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.content.strip()

    def crossover(self, prompt_a: str, prompt_b: str) -> str:
        """Combine strengths of two prompts."""
        user_msg = CROSSOVER_USER_TEMPLATE.format(prompt_a=prompt_a, prompt_b=prompt_b)
        resp = self.llm.chat(
            system=MUTATION_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.content.strip()

    # ── helpers ──────────────────────────────────────────────

    def _parse_variants(self, text: str, fallback_prompt: str, expected: int) -> list[str]:
        """Parse a JSON array of prompt strings from the LLM response."""
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            variants = json.loads(text)
            if isinstance(variants, list):
                return [str(v).strip() for v in variants if str(v).strip()][:expected]
        except json.JSONDecodeError:
            logger.warning("Failed to parse seed variants JSON, falling back to individual mutations")

        # Fallback: generate variants one-by-one
        results = []
        for _ in range(expected):
            try:
                results.append(self.mutate(fallback_prompt))
            except Exception as exc:
                logger.warning("Mutation call failed: %s", exc)
        return results
