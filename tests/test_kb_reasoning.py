"""Tests for evidence ranking and contradiction detection."""

from ct.kb.reasoning import EvidenceReasoner
from ct.kb.substrate import KnowledgeSubstrate


def test_detect_contradictions(tmp_path):
    substrate = KnowledgeSubstrate(path=tmp_path / "substrate.json")
    gene = substrate.upsert_entity(entity_type="gene", name="TP53")
    disease = substrate.upsert_entity(entity_type="disease", name="AML")

    ev1 = substrate.add_evidence(source_type="pubmed", source_ref="PMID:1", summary="supports", score=0.9)
    ev2 = substrate.add_evidence(source_type="pubmed", source_ref="PMID:2", summary="contradicts", score=0.8)
    substrate.link_entities(
        subject_id=gene.id,
        predicate="associated_with_disease",
        object_id=disease.id,
        evidence_id=ev1.id,
        polarity="support",
        score=0.9,
    )
    substrate.link_entities(
        subject_id=gene.id,
        predicate="associated_with_disease",
        object_id=disease.id,
        evidence_id=ev2.id,
        polarity="contradict",
        score=0.8,
    )

    reasoner = EvidenceReasoner(substrate)
    contradictions = reasoner.detect_contradictions(entity_id=gene.id)
    assert len(contradictions) == 1
    assert contradictions[0]["support_claims"] == 1
    assert contradictions[0]["contradict_claims"] == 1


def test_rank_relations_prefers_stronger_source(tmp_path):
    substrate = KnowledgeSubstrate(path=tmp_path / "substrate.json")
    gene = substrate.upsert_entity(entity_type="gene", name="TP53")
    d1 = substrate.upsert_entity(entity_type="disease", name="AML")
    d2 = substrate.upsert_entity(entity_type="disease", name="DLBCL")

    ev_pubmed = substrate.add_evidence(source_type="pubmed", source_ref="PMID:1", summary="strong", score=0.8)
    ev_session = substrate.add_evidence(source_type="session", source_ref="s1", summary="weak", score=0.8)
    substrate.link_entities(
        subject_id=gene.id,
        predicate="associated_with_disease",
        object_id=d1.id,
        evidence_id=ev_pubmed.id,
        polarity="support",
        score=0.8,
    )
    substrate.link_entities(
        subject_id=gene.id,
        predicate="associated_with_disease",
        object_id=d2.id,
        evidence_id=ev_session.id,
        polarity="support",
        score=0.8,
    )

    reasoner = EvidenceReasoner(substrate)
    ranked = reasoner.rank_relations(entity_id=gene.id)
    assert len(ranked) == 2
    assert ranked[0]["object_id"] == d1.id
