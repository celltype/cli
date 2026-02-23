"""
Evidence ranking and contradiction analysis over the knowledge substrate.
"""

from __future__ import annotations

import math
import time
from typing import Any

from ct.kb.substrate import KBClaim, KBRelation, KnowledgeSubstrate


SOURCE_WEIGHTS = {
    "pubmed": 0.9,
    "openalex": 0.85,
    "opentargets": 0.92,
    "tool": 0.75,
    "session": 0.65,
    "unknown": 0.5,
}


def _recency_weight(ts: float, now: float) -> float:
    age_days = max((now - ts) / 86400.0, 0.0)
    return math.exp(-age_days / 365.0)


class EvidenceReasoner:
    """Ranking and contradiction detector."""

    def __init__(self, substrate: KnowledgeSubstrate):
        self.substrate = substrate

    def relation_score(self, relation: KBRelation, *, now: float | None = None) -> float:
        """Aggregate weighted confidence for relation claims."""
        now = now or time.time()
        if not relation.claims:
            return 0.0
        weighted = []
        for claim in relation.claims:
            evidence = self.substrate.get_evidence(claim.evidence_id)
            source_weight = SOURCE_WEIGHTS.get(
                (evidence.source_type if evidence else "unknown"),
                SOURCE_WEIGHTS["unknown"],
            )
            recency = _recency_weight(claim.timestamp, now)
            polarity = 1.0 if claim.polarity == "support" else (-1.0 if claim.polarity == "contradict" else 0.2)
            score = claim.score * source_weight * recency * polarity
            weighted.append(score)
        return sum(weighted) / max(len(weighted), 1)

    def rank_relations(
        self,
        *,
        entity_id: str | None = None,
        predicate: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return highest-confidence relations."""
        rows = []
        for relation in self.substrate.list_relations():
            if entity_id and relation.subject_id != entity_id and relation.object_id != entity_id:
                continue
            if predicate and relation.predicate != predicate:
                continue
            score = self.relation_score(relation)
            rows.append(
                {
                    "relation_id": relation.id,
                    "subject_id": relation.subject_id,
                    "predicate": relation.predicate,
                    "object_id": relation.object_id,
                    "score": round(score, 4),
                    "n_claims": len(relation.claims),
                    "last_seen": relation.last_seen,
                }
            )
        rows.sort(key=lambda r: (r["score"], r["n_claims"], r["last_seen"]), reverse=True)
        return rows[: max(limit, 0)]

    def detect_contradictions(
        self,
        *,
        entity_id: str | None = None,
        predicate: str | None = None,
        min_claims: int = 2,
    ) -> list[dict[str, Any]]:
        """Find relations with mixed support and contradiction evidence."""
        contradictions = []
        for relation in self.substrate.list_relations():
            if entity_id and relation.subject_id != entity_id and relation.object_id != entity_id:
                continue
            if predicate and relation.predicate != predicate:
                continue
            if len(relation.claims) < min_claims:
                continue
            support = [c for c in relation.claims if c.polarity == "support"]
            contradict = [c for c in relation.claims if c.polarity == "contradict"]
            if not support or not contradict:
                continue
            contradictions.append(
                {
                    "relation_id": relation.id,
                    "subject_id": relation.subject_id,
                    "predicate": relation.predicate,
                    "object_id": relation.object_id,
                    "support_claims": len(support),
                    "contradict_claims": len(contradict),
                    "support_score": round(self._avg_claim_score(support), 4),
                    "contradict_score": round(self._avg_claim_score(contradict), 4),
                    "last_seen": relation.last_seen,
                }
            )
        contradictions.sort(
            key=lambda c: (
                min(c["support_claims"], c["contradict_claims"]),
                max(c["support_score"], c["contradict_score"]),
                c["last_seen"],
            ),
            reverse=True,
        )
        return contradictions

    @staticmethod
    def _avg_claim_score(claims: list[KBClaim]) -> float:
        if not claims:
            return 0.0
        return sum(c.score for c in claims) / len(claims)
