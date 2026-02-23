"""Tests for knowledge ingestion pipeline."""

import json

from ct.kb.ingest import KnowledgeIngestionPipeline
from ct.kb.substrate import KnowledgeSubstrate


def test_ingest_evidence_store(tmp_path):
    evidence_path = tmp_path / "evidence.jsonl"
    record = {
        "session_id": "sess-1",
        "query": "Profile TP53 in AML",
        "n_completed_steps": 1,
        "steps": [
            {
                "id": 1,
                "tool": "literature.pubmed_search",
                "description": "Search papers",
                "result_summary": "Found TP53 AML evidence",
            }
        ],
        "synthesis_preview": "TP53 appears relevant in AML.",
    }
    evidence_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    substrate = KnowledgeSubstrate(path=tmp_path / "substrate.json")
    pipeline = KnowledgeIngestionPipeline(
        substrate=substrate,
        state_path=tmp_path / "state.json",
        evidence_path=evidence_path,
    )
    out = pipeline.ingest_evidence_store()
    assert out["ingested_records"] == 1
    summary = substrate.summary()
    assert summary["n_entities"] > 0
    assert summary["n_relations"] > 0
    assert summary["n_evidence"] > 0


def test_ingest_pubmed_with_mock(monkeypatch, tmp_path):
    def fake_pubmed_search(query: str, max_results: int = 10, **kwargs):
        return {
            "summary": "ok",
            "articles": [
                {"pmid": "123456", "title": "TP53 in AML", "journal": "J Clin", "publication_year": 2024},
            ],
        }

    monkeypatch.setattr("ct.tools.literature.pubmed_search", fake_pubmed_search)

    substrate = KnowledgeSubstrate(path=tmp_path / "substrate.json")
    pipeline = KnowledgeIngestionPipeline(
        substrate=substrate,
        state_path=tmp_path / "state.json",
        evidence_path=tmp_path / "empty.jsonl",
    )
    out = pipeline.ingest_pubmed(query="TP53 AML", max_results=1)
    assert out["ingested_articles"] == 1
    hits = substrate.search_entities("PMID:123456")
    assert len(hits) == 1


def test_ingest_opentargets_with_mock(monkeypatch, tmp_path):
    def fake_ot_search(query: str, entity_type: str = "target", **kwargs):
        return {
            "entity_id": "ENSG00000141510",
            "name": "Tumor protein p53",
            "symbol": "TP53",
            "top_disease_associations": [
                {"disease_name": "acute myeloid leukemia", "overall_score": 0.83},
            ],
        }

    monkeypatch.setattr("ct.tools.data_api.opentargets_search", fake_ot_search)

    substrate = KnowledgeSubstrate(path=tmp_path / "substrate.json")
    pipeline = KnowledgeIngestionPipeline(
        substrate=substrate,
        state_path=tmp_path / "state.json",
        evidence_path=tmp_path / "empty.jsonl",
    )
    out = pipeline.ingest_opentargets(query="TP53")
    assert out["relations_created"] == 1
    related = substrate.related_entities("gene:TP53")
    assert len(related) == 1
