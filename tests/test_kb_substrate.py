"""Tests for canonical knowledge substrate."""

from ct.kb.substrate import KnowledgeSubstrate


def test_upsert_entity_normalizes_gene(tmp_path):
    substrate = KnowledgeSubstrate(path=tmp_path / "substrate.json")
    entity = substrate.upsert_entity(entity_type="gene", name="tp53")
    assert entity.id == "gene:TP53"
    assert entity.name == "tp53"
    substrate.save()

    reloaded = KnowledgeSubstrate(path=tmp_path / "substrate.json")
    found = reloaded.get_entity("gene:TP53")
    assert found is not None
    assert found.id == "gene:TP53"


def test_add_evidence_and_link_relation(tmp_path):
    substrate = KnowledgeSubstrate(path=tmp_path / "substrate.json")
    gene = substrate.upsert_entity(entity_type="gene", name="TP53")
    disease = substrate.upsert_entity(entity_type="disease", name="acute myeloid leukemia")
    ev = substrate.add_evidence(
        source_type="pubmed",
        source_ref="PMID:123",
        summary="TP53 implicated in AML",
        score=0.9,
    )
    rel = substrate.link_entities(
        subject_id=gene.id,
        predicate="associated_with_disease",
        object_id=disease.id,
        evidence_id=ev.id,
        polarity="support",
        score=0.8,
    )
    assert rel.id == f"{gene.id}|associated_with_disease|{disease.id}"
    rows = substrate.related_entities(gene.id)
    assert len(rows) == 1
    assert rows[0]["other_entity_id"] == disease.id
    assert rows[0]["support_claims"] == 1


def test_search_entities_matches_synonyms(tmp_path):
    substrate = KnowledgeSubstrate(path=tmp_path / "substrate.json")
    substrate.upsert_entity(
        entity_type="compound",
        name="lenalidomide",
        synonyms=["CC-5013"],
    )
    hits = substrate.search_entities("cc-5013")
    assert len(hits) == 1
    assert hits[0].name == "lenalidomide"
