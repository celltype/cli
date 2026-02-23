"""
Knowledge ingestion and normalization pipeline.

Builds the knowledge substrate from:
- local evidence logs (always available)
- optional live APIs (PubMed, OpenAlex, Open Targets)
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any

from ct.kb.substrate import KnowledgeSubstrate


class KnowledgeIngestionPipeline:
    """Incremental ingestion pipeline into the canonical knowledge substrate."""

    def __init__(
        self,
        substrate: KnowledgeSubstrate | None = None,
        *,
        state_path: Path | None = None,
        evidence_path: Path | None = None,
    ):
        self.substrate = substrate or KnowledgeSubstrate()
        self.state_path = state_path or (Path.home() / ".ct" / "knowledge" / "ingest_state.json")
        self.evidence_path = evidence_path or (Path.home() / ".ct" / "evidence" / "evidence.jsonl")
        self._state = self._load_state()

    def _default_state(self) -> dict[str, Any]:
        return {
            "updated_at": time.time(),
            "evidence_line_offset": 0,
            "source_runs": {},
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._default_state()
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._default_state()
        if not isinstance(data, dict):
            return self._default_state()
        data.setdefault("updated_at", time.time())
        data.setdefault("evidence_line_offset", 0)
        data.setdefault("source_runs", {})
        return data

    def save_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state["updated_at"] = time.time()
        self.state_path.write_text(
            json.dumps(self._state, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def ingest(
        self,
        *,
        source: str,
        query: str | None = None,
        max_results: int = 10,
        scan_limit: int = 1000,
    ) -> dict[str, Any]:
        """Ingest from one source into substrate."""
        src = (source or "").strip().lower()
        if src == "evidence_store":
            return self.ingest_evidence_store(scan_limit=scan_limit)
        if src == "pubmed":
            if not query:
                return {"error": "query is required for source=pubmed"}
            return self.ingest_pubmed(query=query, max_results=max_results)
        if src == "openalex":
            if not query:
                return {"error": "query is required for source=openalex"}
            return self.ingest_openalex(query=query, max_results=max_results)
        if src == "opentargets":
            if not query:
                return {"error": "query is required for source=opentargets"}
            return self.ingest_opentargets(query=query)
        return {"error": f"Unknown source '{source}'"}

    def ingest_evidence_store(self, *, scan_limit: int = 1000) -> dict[str, Any]:
        """Ingest new rows from local evidence log."""
        if not self.evidence_path.exists():
            return {
                "summary": "No local evidence store found.",
                "source": "evidence_store",
                "ingested_records": 0,
            }

        try:
            lines = self.evidence_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            return {"error": f"Failed reading evidence store: {exc}"}

        start = int(self._state.get("evidence_line_offset", 0))
        if start >= len(lines):
            return {
                "summary": "No new evidence records to ingest.",
                "source": "evidence_store",
                "ingested_records": 0,
            }

        new_lines = lines[start:][: max(scan_limit, 0)]
        ingested = 0
        linked_entities = 0
        for line in new_lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            entities = self._ingest_evidence_record(record)
            linked_entities += entities
            ingested += 1

        self._state["evidence_line_offset"] = start + len(new_lines)
        self._state["source_runs"]["evidence_store"] = time.time()
        self.save_state()
        self.substrate.save()
        return {
            "summary": (
                f"Ingested {ingested} evidence record(s) from local store; "
                f"linked {linked_entities} entity mention(s)."
            ),
            "source": "evidence_store",
            "ingested_records": ingested,
            "linked_entities": linked_entities,
            "new_offset": self._state["evidence_line_offset"],
        }

    def _ingest_evidence_record(self, record: dict[str, Any]) -> int:
        query = str(record.get("query", "")).strip()
        synthesis = str(record.get("synthesis_preview", "")).strip()
        session_id = str(record.get("session_id", "")).strip()
        steps = record.get("steps", []) or []

        query_entities = self._extract_entity_mentions(query)
        if not query_entities and query:
            query_entities = [query]

        ev = self.substrate.add_evidence(
            source_type="session",
            source_ref=session_id or "unknown_session",
            summary=synthesis or query,
            score=0.6,
            tags=["session", "evidence_store"],
            metadata={"n_completed_steps": int(record.get("n_completed_steps", 0) or 0)},
        )

        entity_ids = []
        for mention in query_entities:
            entity_type = KnowledgeSubstrate.infer_entity_type(mention)
            entity = self.substrate.upsert_entity(entity_type=entity_type, name=mention)
            entity_ids.append(entity.id)

        for i, left in enumerate(entity_ids):
            for right in entity_ids[i + 1:]:
                self.substrate.link_entities(
                    subject_id=left,
                    predicate="co_mentioned_in_query",
                    object_id=right,
                    evidence_id=ev.id,
                    polarity="support",
                    score=0.55,
                    metadata={"source": "query"},
                )

        linked = len(entity_ids)
        for step in steps:
            tool_name = str(step.get("tool", "")).strip()
            step_desc = str(step.get("description", "")).strip()
            step_summary = str(step.get("result_summary", "")).strip()
            if not tool_name:
                continue

            tool_entity = self.substrate.upsert_entity(
                entity_type="tool",
                name=tool_name,
                identifier=tool_name,
            )
            step_ev = self.substrate.add_evidence(
                source_type="tool",
                source_ref=tool_name,
                summary=(step_summary or step_desc)[:1200],
                score=0.65,
                tags=["step_result"],
                metadata={"step_id": step.get("id"), "session_id": session_id},
            )

            for eid in entity_ids:
                self.substrate.link_entities(
                    subject_id=eid,
                    predicate="analyzed_with",
                    object_id=tool_entity.id,
                    evidence_id=step_ev.id,
                    polarity="support",
                    score=0.65,
                    metadata={"step_id": step.get("id")},
                )

            step_entities = self._extract_entity_mentions(f"{step_desc} {step_summary}")
            for mention in step_entities:
                if mention in query_entities:
                    continue
                se = self.substrate.upsert_entity(
                    entity_type=KnowledgeSubstrate.infer_entity_type(mention),
                    name=mention,
                )
                linked += 1
                for eid in entity_ids:
                    self.substrate.link_entities(
                        subject_id=eid,
                        predicate="associated_with",
                        object_id=se.id,
                        evidence_id=step_ev.id,
                        polarity="support",
                        score=0.6,
                    )
        return linked

    def ingest_pubmed(self, *, query: str, max_results: int = 10) -> dict[str, Any]:
        from ct.tools.literature import pubmed_search

        result = pubmed_search(query=query, max_results=max_results)
        if result.get("error"):
            return {"error": result["error"], "source": "pubmed"}

        articles = result.get("articles", []) or []
        query_entities = self._get_or_create_query_entities(query)
        n_links = 0
        for art in articles:
            pmid = str(art.get("pmid", "")).strip()
            title = str(art.get("title", "")).strip()
            if not pmid:
                continue
            pub = self.substrate.upsert_entity(
                entity_type="publication",
                name=title or f"PMID {pmid}",
                identifier=f"PMID:{pmid}",
                metadata={"pmid": pmid, "journal": art.get("journal", "")},
            )
            ev = self.substrate.add_evidence(
                source_type="pubmed",
                source_ref=f"PMID:{pmid}",
                summary=title,
                score=0.75,
                tags=["literature"],
                metadata={"year": art.get("publication_year")},
            )
            for eid in query_entities:
                self.substrate.link_entities(
                    subject_id=eid,
                    predicate="supported_by_literature",
                    object_id=pub.id,
                    evidence_id=ev.id,
                    polarity="support",
                    score=0.75,
                )
                n_links += 1
        self._state["source_runs"]["pubmed"] = time.time()
        self.save_state()
        self.substrate.save()
        return {
            "summary": f"Ingested {len(articles)} PubMed article(s) for '{query}'.",
            "source": "pubmed",
            "ingested_articles": len(articles),
            "links_created": n_links,
        }

    def ingest_openalex(self, *, query: str, max_results: int = 10) -> dict[str, Any]:
        from ct.tools.literature import openalex_search

        result = openalex_search(query=query, max_results=max_results)
        if result.get("error"):
            return {"error": result["error"], "source": "openalex"}

        articles = result.get("articles", []) or []
        query_entities = self._get_or_create_query_entities(query)
        n_links = 0
        for art in articles:
            doi = str(art.get("doi", "")).strip()
            title = str(art.get("title", "")).strip()
            if not doi and not title:
                continue
            pub_id = doi or title
            pub = self.substrate.upsert_entity(
                entity_type="publication",
                name=title or pub_id,
                identifier=pub_id,
                metadata={
                    "doi": doi,
                    "source": art.get("source", ""),
                    "year": art.get("publication_year"),
                    "cited_by_count": art.get("cited_by_count", 0),
                },
            )
            ev = self.substrate.add_evidence(
                source_type="openalex",
                source_ref=pub_id,
                summary=title,
                score=0.72,
                tags=["literature"],
            )
            for eid in query_entities:
                self.substrate.link_entities(
                    subject_id=eid,
                    predicate="supported_by_literature",
                    object_id=pub.id,
                    evidence_id=ev.id,
                    polarity="support",
                    score=0.72,
                )
                n_links += 1
        self._state["source_runs"]["openalex"] = time.time()
        self.save_state()
        self.substrate.save()
        return {
            "summary": f"Ingested {len(articles)} OpenAlex work(s) for '{query}'.",
            "source": "openalex",
            "ingested_works": len(articles),
            "links_created": n_links,
        }

    def ingest_opentargets(self, *, query: str) -> dict[str, Any]:
        from ct.tools.data_api import opentargets_search

        result = opentargets_search(query=query, entity_type="target")
        if result.get("error"):
            return {"error": result["error"], "source": "opentargets"}

        target_name = str(result.get("name", query)).strip() or query
        target_symbol = str(result.get("symbol", "")).strip()
        target_key = target_symbol or target_name
        target = self.substrate.upsert_entity(
            entity_type="gene",
            name=target_name,
            identifier=target_key,
            synonyms=[target_symbol] if target_symbol else [],
            metadata={"opentargets_id": result.get("entity_id", "")},
        )

        associations = result.get("top_disease_associations", []) or result.get("associations", []) or []
        created = 0
        for assoc in associations[:20]:
            disease_name = str(assoc.get("disease_name") or assoc.get("disease", "")).strip()
            if not disease_name:
                continue
            disease = self.substrate.upsert_entity(entity_type="disease", name=disease_name)
            score = float(assoc.get("overall_score", 0.5) or 0.5)
            ev = self.substrate.add_evidence(
                source_type="opentargets",
                source_ref=str(result.get("entity_id", "")),
                summary=f"{target_name} association with {disease_name}",
                score=max(0.4, min(score, 1.0)),
                tags=["genetics", "target_disease"],
                metadata={"association_score": score},
            )
            self.substrate.link_entities(
                subject_id=target.id,
                predicate="associated_with_disease",
                object_id=disease.id,
                evidence_id=ev.id,
                polarity="support",
                score=max(0.4, min(score, 1.0)),
                metadata={"source": "opentargets"},
            )
            created += 1

        self._state["source_runs"]["opentargets"] = time.time()
        self.save_state()
        self.substrate.save()
        return {
            "summary": f"Ingested Open Targets associations for '{query}' ({created} relation(s)).",
            "source": "opentargets",
            "relations_created": created,
        }

    def _get_or_create_query_entities(self, query: str) -> list[str]:
        mentions = self._extract_entity_mentions(query)
        if not mentions and query:
            mentions = [query]
        ids = []
        for mention in mentions:
            entity = self.substrate.upsert_entity(
                entity_type=KnowledgeSubstrate.infer_entity_type(mention),
                name=mention,
            )
            ids.append(entity.id)
        return ids

    def _extract_entity_mentions(self, text: str) -> list[str]:
        mentions = []
        # Add PMID/NCT mentions if present.
        mentions.extend(re.findall(r"\bPMID[:\s]?\d+\b", text or "", flags=re.IGNORECASE))
        mentions.extend(re.findall(r"\bNCT\d{8}\b", text or "", flags=re.IGNORECASE))
        dedup = []
        seen = set()
        for m in mentions:
            norm = m.strip()
            if not norm:
                continue
            key = norm.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(norm)
        return dedup
