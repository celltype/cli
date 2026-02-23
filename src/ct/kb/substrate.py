"""
Canonical R&D knowledge substrate.

Stores normalized entities, evidence, and typed relations in a local JSON store.
This is the foundational layer for cross-modal pharma knowledge accumulation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import time
from typing import Any


_GENE_RE = re.compile(r"^[A-Z][A-Z0-9-]{1,9}$")


@dataclass
class KBEntity:
    id: str
    entity_type: str
    name: str
    synonyms: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


@dataclass
class KBEvidence:
    id: str
    source_type: str
    source_ref: str
    summary: str
    score: float = 0.5
    tags: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KBClaim:
    evidence_id: str
    polarity: str = "support"  # support | contradict | neutral
    score: float = 0.5
    timestamp: float = field(default_factory=time.time)


@dataclass
class KBRelation:
    id: str
    subject_id: str
    predicate: str
    object_id: str
    claims: list[KBClaim] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


class KnowledgeSubstrate:
    """Local persistent knowledge substrate."""

    SCHEMA_VERSION = 1

    def __init__(self, path: Path | None = None):
        self.path = path or (Path.home() / ".ct" / "knowledge" / "substrate.json")
        self._data = self._load()

    @staticmethod
    def normalize_identifier(entity_type: str, value: str) -> str:
        """Normalize external identifiers into stable canonical keys."""
        et = (entity_type or "unknown").strip().lower()
        raw = (value or "").strip()
        if not raw:
            raw = "unknown"

        if et == "gene":
            norm = re.sub(r"[^A-Za-z0-9-]", "", raw).upper()
            return norm or "UNKNOWN"
        if et in {"disease", "indication", "pathway", "phenotype"}:
            norm = re.sub(r"\s+", " ", raw.lower()).strip()
            return norm or "unknown"
        if et in {"compound", "drug"}:
            norm = re.sub(r"\s+", " ", raw).strip()
            return norm
        if et in {"publication", "trial"}:
            return raw.upper()
        return re.sub(r"\s+", " ", raw).strip()

    @staticmethod
    def infer_entity_type(text: str) -> str:
        """Infer coarse entity type from surface form."""
        token = (text or "").strip()
        if not token:
            return "unknown"
        if token.upper().startswith("PMID"):
            return "publication"
        if token.upper().startswith("NCT") and token[3:].isdigit():
            return "trial"
        if _GENE_RE.match(token):
            return "gene"
        if any(c.isdigit() for c in token) and "-" in token:
            return "compound"
        if len(token.split()) >= 2:
            return "disease"
        return "unknown"

    def _default(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "created_at": time.time(),
            "updated_at": time.time(),
            "next_evidence_id": 1,
            "entities": {},
            "evidence": {},
            "relations": {},
        }

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._default()

        if not isinstance(payload, dict):
            return self._default()
        payload.setdefault("schema_version", self.SCHEMA_VERSION)
        payload.setdefault("created_at", time.time())
        payload.setdefault("updated_at", time.time())
        payload.setdefault("next_evidence_id", 1)
        payload.setdefault("entities", {})
        payload.setdefault("evidence", {})
        payload.setdefault("relations", {})
        return payload

    def save(self):
        """Persist substrate to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data["updated_at"] = time.time()
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def upsert_entity(
        self,
        *,
        entity_type: str,
        name: str,
        identifier: str | None = None,
        synonyms: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KBEntity:
        """Insert/update entity and return canonical record."""
        canonical = self.normalize_identifier(entity_type, identifier or name)
        entity_id = f"{entity_type.lower()}:{canonical}"
        now = time.time()
        existing = self._data["entities"].get(entity_id)
        syn_items = list(synonyms or []) + [name]
        if identifier:
            syn_items.append(str(identifier))
        syn = sorted(set(syn_items))
        if existing:
            existing_syn = set(existing.get("synonyms", []))
            existing["synonyms"] = sorted(existing_syn | set(syn))
            if metadata:
                existing_meta = existing.get("metadata", {})
                existing_meta.update(metadata)
                existing["metadata"] = existing_meta
            existing["last_seen"] = now
            self._data["entities"][entity_id] = existing
            return KBEntity(**existing)

        entity = KBEntity(
            id=entity_id,
            entity_type=entity_type.lower(),
            name=name,
            synonyms=syn,
            metadata=metadata or {},
            first_seen=now,
            last_seen=now,
        )
        self._data["entities"][entity_id] = asdict(entity)
        return entity

    def get_entity(self, entity_id: str) -> KBEntity | None:
        rec = self._data["entities"].get(entity_id)
        if not rec:
            return None
        return KBEntity(**rec)

    def add_evidence(
        self,
        *,
        source_type: str,
        source_ref: str,
        summary: str,
        score: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KBEvidence:
        """Record evidence statement with provenance."""
        next_id = int(self._data.get("next_evidence_id", 1))
        evidence_id = f"e:{next_id}"
        self._data["next_evidence_id"] = next_id + 1
        ev = KBEvidence(
            id=evidence_id,
            source_type=(source_type or "unknown").lower(),
            source_ref=source_ref or "",
            summary=(summary or "")[:2000],
            score=max(0.0, min(1.0, float(score))),
            tags=tags or [],
            metadata=metadata or {},
        )
        self._data["evidence"][evidence_id] = asdict(ev)
        return ev

    def get_evidence(self, evidence_id: str) -> KBEvidence | None:
        rec = self._data["evidence"].get(evidence_id)
        if not rec:
            return None
        return KBEvidence(**rec)

    def link_entities(
        self,
        *,
        subject_id: str,
        predicate: str,
        object_id: str,
        evidence_id: str,
        polarity: str = "support",
        score: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> KBRelation:
        """Create or update relation edge with evidence claim."""
        relation_id = f"{subject_id}|{predicate}|{object_id}"
        now = time.time()
        claim = KBClaim(
            evidence_id=evidence_id,
            polarity=polarity if polarity in {"support", "contradict", "neutral"} else "neutral",
            score=max(0.0, min(1.0, float(score))),
            timestamp=now,
        )
        existing = self._data["relations"].get(relation_id)
        if existing:
            existing_claims = [
                KBClaim(**c) if isinstance(c, dict) else c for c in existing.get("claims", [])
            ]
            existing_claims.append(claim)
            existing["claims"] = [asdict(c) for c in existing_claims]
            if metadata:
                existing_meta = existing.get("metadata", {})
                existing_meta.update(metadata)
                existing["metadata"] = existing_meta
            existing["last_seen"] = now
            self._data["relations"][relation_id] = existing
            return KBRelation(
                id=relation_id,
                subject_id=existing["subject_id"],
                predicate=existing["predicate"],
                object_id=existing["object_id"],
                claims=[KBClaim(**c) for c in existing["claims"]],
                metadata=existing.get("metadata", {}),
                first_seen=float(existing.get("first_seen", now)),
                last_seen=float(existing.get("last_seen", now)),
            )

        rel = KBRelation(
            id=relation_id,
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            claims=[claim],
            metadata=metadata or {},
            first_seen=now,
            last_seen=now,
        )
        self._data["relations"][relation_id] = {
            "id": rel.id,
            "subject_id": rel.subject_id,
            "predicate": rel.predicate,
            "object_id": rel.object_id,
            "claims": [asdict(claim)],
            "metadata": rel.metadata,
            "first_seen": rel.first_seen,
            "last_seen": rel.last_seen,
        }
        return rel

    def search_entities(self, query: str, limit: int = 20) -> list[KBEntity]:
        """Simple text search by canonical name/synonyms."""
        q = (query or "").strip().lower()
        if not q:
            return []
        hits: list[tuple[float, KBEntity]] = []
        terms = set(re.findall(r"[a-z0-9-]{2,}", q))
        for rec in self._data["entities"].values():
            entity = KBEntity(**rec)
            haystack = " ".join([entity.name] + entity.synonyms).lower()
            if q in haystack:
                score = 1.0
            else:
                tokens = set(re.findall(r"[a-z0-9-]{2,}", haystack))
                score = len(terms & tokens) / max(len(terms), 1)
            if score <= 0:
                continue
            hits.append((score, entity))
        hits.sort(key=lambda x: (x[0], x[1].last_seen), reverse=True)
        return [h[1] for h in hits[: max(limit, 0)]]

    def related_entities(
        self,
        entity_id: str,
        *,
        predicate: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List entities connected via relations with aggregate claim stats."""
        rows = []
        for rel in self._data["relations"].values():
            if rel.get("subject_id") != entity_id and rel.get("object_id") != entity_id:
                continue
            if predicate and rel.get("predicate") != predicate:
                continue
            claims = [KBClaim(**c) for c in rel.get("claims", [])]
            support = sum(1 for c in claims if c.polarity == "support")
            contradict = sum(1 for c in claims if c.polarity == "contradict")
            neutral = sum(1 for c in claims if c.polarity == "neutral")
            avg_score = sum(c.score for c in claims) / max(len(claims), 1)
            other = rel["object_id"] if rel["subject_id"] == entity_id else rel["subject_id"]
            rows.append(
                {
                    "relation_id": rel["id"],
                    "predicate": rel["predicate"],
                    "other_entity_id": other,
                    "support_claims": support,
                    "contradict_claims": contradict,
                    "neutral_claims": neutral,
                    "claim_count": len(claims),
                    "average_claim_score": round(avg_score, 4),
                    "last_seen": float(rel.get("last_seen", 0)),
                }
            )
        rows.sort(
            key=lambda r: (r["support_claims"] - r["contradict_claims"], r["average_claim_score"], r["last_seen"]),
            reverse=True,
        )
        return rows[: max(limit, 0)]

    def list_relations(self) -> list[KBRelation]:
        items = []
        for rec in self._data["relations"].values():
            items.append(
                KBRelation(
                    id=rec["id"],
                    subject_id=rec["subject_id"],
                    predicate=rec["predicate"],
                    object_id=rec["object_id"],
                    claims=[KBClaim(**c) for c in rec.get("claims", [])],
                    metadata=rec.get("metadata", {}),
                    first_seen=float(rec.get("first_seen", 0)),
                    last_seen=float(rec.get("last_seen", 0)),
                )
            )
        return items

    def summary(self) -> dict[str, Any]:
        """High-level substrate stats."""
        entity_types: dict[str, int] = {}
        for rec in self._data["entities"].values():
            et = rec.get("entity_type", "unknown")
            entity_types[et] = entity_types.get(et, 0) + 1
        return {
            "path": str(self.path),
            "schema_version": self._data.get("schema_version", self.SCHEMA_VERSION),
            "n_entities": len(self._data["entities"]),
            "n_relations": len(self._data["relations"]),
            "n_evidence": len(self._data["evidence"]),
            "entity_types": entity_types,
            "updated_at": self._data.get("updated_at"),
        }
