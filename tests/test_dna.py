"""Tests for DNA biology tools."""

from __future__ import annotations

from ct.tools.dna import (
    reverse_complement,
    translate,
    find_orfs,
    codon_optimize,
    restriction_sites,
    virtual_digest,
    primer_design,
    pcr_protocol,
    gibson_design,
    golden_gate_design,
)


def test_reverse_complement():
    result = reverse_complement("ATGC")
    assert result["reverse_complement"] == "GCAT"


def test_translate_basic():
    result = translate("ATGGAACTGTAA")
    assert result["protein"].startswith("MEL")


def test_find_orfs_detects_orf():
    result = find_orfs("CCCATGAAACCCGGGTAAAGG", min_aa_length=3)
    assert result["count"] >= 1


def test_codon_optimize_human():
    result = codon_optimize("MKT", species="human")
    assert "optimized_dna" in result
    assert len(result["optimized_dna"]) == 9


def test_restriction_sites_ecori():
    seq = "AAAGAATTCTTTGAATTC"
    result = restriction_sites(seq, enzymes=["EcoRI"])
    assert result["results"][0]["n_sites"] == 2


def test_virtual_digest_linear():
    seq = "AAAAAGAATTCAAAA"
    result = virtual_digest(seq, enzymes=["EcoRI"], circular=False)
    assert result["n_fragments"] == 2


def test_primer_design_returns_pair():
    seq = "ATG" + ("ACGT" * 80) + "TAA"
    result = primer_design(seq, target_start=50, target_end=250)
    assert "forward_primer" in result
    assert "reverse_primer" in result


def test_pcr_protocol():
    result = pcr_protocol(product_size_bp=1500, primer_tm=62.0)
    assert result["extension_time_s"] >= 30
    assert result["cycles"] == 30


def test_gibson_design_requires_multiple_fragments():
    bad = gibson_design(["ATGC"])
    assert bad["error"] == "invalid_fragments"


def test_gibson_and_golden_gate_success():
    gibson = gibson_design(["ATGCGTACGTACGTACGTACGTACG", "CGTACGTACGTACGTACGTACGTAA"])
    assert "joins" in gibson

    gg = golden_gate_design(["promoter", "cds", "terminator"])
    assert len(gg["joins"]) == 2
