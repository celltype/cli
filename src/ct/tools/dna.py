"""
DNA biology utilities for sequence analysis and planning.

Includes sequence transforms, ORF detection, primer suggestions, codon optimization,
restriction analysis, and assembly helper templates.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from ct.tools import registry


_DNA_ALPHABET = set("ACGTN")
_STOP_CODONS = {"TAA", "TAG", "TGA"}
_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Preferred codons (lightweight, pragmatic defaults).
_PREF_CODONS = {
    "human": {
        "A": "GCC", "R": "CGC", "N": "AAC", "D": "GAC", "C": "TGC", "Q": "CAG", "E": "GAG", "G": "GGC",
        "H": "CAC", "I": "ATC", "L": "CTG", "K": "AAG", "M": "ATG", "F": "TTC", "P": "CCC", "S": "AGC",
        "T": "ACC", "W": "TGG", "Y": "TAC", "V": "GTG",
    },
    "ecoli": {
        "A": "GCG", "R": "CGT", "N": "AAC", "D": "GAT", "C": "TGC", "Q": "CAG", "E": "GAA", "G": "GGC",
        "H": "CAT", "I": "ATT", "L": "CTG", "K": "AAA", "M": "ATG", "F": "TTT", "P": "CCG", "S": "TCT",
        "T": "ACC", "W": "TGG", "Y": "TAT", "V": "GTG",
    },
}

_ENZYME_MOTIFS = {
    "EcoRI": "GAATTC",
    "BamHI": "GGATCC",
    "HindIII": "AAGCTT",
    "NotI": "GCGGCCGC",
    "XhoI": "CTCGAG",
    "NheI": "GCTAGC",
    "BsaI": "GGTCTC",
    "BsmBI": "CGTCTC",
}


@dataclass
class _PrimerCandidate:
    seq: str
    tm: float
    gc: float


def _clean_seq(seq: str) -> str:
    return re.sub(r"\s+", "", str(seq or "").upper())


def _validate_dna(seq: str) -> tuple[str, str | None]:
    s = _clean_seq(seq)
    if not s:
        return s, "sequence is required"
    if any(base not in _DNA_ALPHABET for base in s):
        return s, "sequence contains non-DNA characters"
    return s, None


def _reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTN", "TGCAN")
    return seq.translate(table)[::-1]


def _wallace_tm(seq: str) -> float:
    seq = seq.upper()
    at = seq.count("A") + seq.count("T")
    gc = seq.count("G") + seq.count("C")
    return 2.0 * at + 4.0 * gc


def _gc_content(seq: str) -> float:
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return (100.0 * gc / len(seq)) if seq else 0.0


def _translate_dna(seq: str, frame: int = 1, to_stop: bool = False) -> str:
    offset = max(0, min(2, frame - 1))
    aa = []
    for i in range(offset, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        residue = _CODON_TABLE.get(codon, "X")
        if residue == "*" and to_stop:
            break
        aa.append(residue)
    return "".join(aa)


def _find_sites(seq: str, motif: str) -> list[int]:
    positions = []
    start = 0
    while True:
        idx = seq.find(motif, start)
        if idx < 0:
            break
        positions.append(idx + 1)  # 1-based
        start = idx + 1
    return positions


def _pick_primer_candidates(seq: str, min_len: int, max_len: int, tm_target: float) -> list[_PrimerCandidate]:
    out = []
    for n in range(min_len, max_len + 1):
        if n > len(seq):
            break
        cand = seq[:n]
        tm = _wallace_tm(cand)
        gc = _gc_content(cand)
        if 35 <= gc <= 70:
            out.append(_PrimerCandidate(seq=cand, tm=tm, gc=gc))
    out.sort(key=lambda c: abs(c.tm - tm_target))
    return out


@registry.register(
    name="dna.reverse_complement",
    description="Compute reverse complement of a DNA sequence",
    category="dna",
    parameters={"sequence": "DNA sequence"},
    usage_guide="Use for strand conversion and antisense oligo planning.",
)
def reverse_complement(sequence: str, **kwargs) -> dict:
    seq, err = _validate_dna(sequence)
    if err:
        return {"summary": err, "error": "invalid_sequence"}
    rc = _reverse_complement(seq)
    return {"summary": f"Reverse complement computed ({len(seq)} bp).", "sequence": seq, "reverse_complement": rc}


@registry.register(
    name="dna.translate",
    description="Translate DNA sequence to amino-acid sequence",
    category="dna",
    parameters={
        "sequence": "DNA sequence",
        "frame": "Reading frame 1-3 (default 1)",
        "to_stop": "Stop translation at first stop codon (default false)",
    },
    usage_guide="Use to inspect coding potential and validate ORF translations.",
)
def translate(sequence: str, frame: int = 1, to_stop: bool = False, **kwargs) -> dict:
    seq, err = _validate_dna(sequence)
    if err:
        return {"summary": err, "error": "invalid_sequence"}
    frame = max(1, min(3, int(frame)))
    aa = _translate_dna(seq, frame=frame, to_stop=bool(to_stop))
    return {
        "summary": f"Translated DNA in frame {frame}: {len(aa)} aa.",
        "frame": frame,
        "protein": aa,
        "protein_length": len(aa),
    }


@registry.register(
    name="dna.find_orfs",
    description="Find open reading frames in a DNA sequence",
    category="dna",
    parameters={
        "sequence": "DNA sequence",
        "min_aa_length": "Minimum amino-acid length (default 30)",
        "include_reverse": "Also scan reverse complement (default false)",
    },
    usage_guide="Use to identify candidate coding regions before cloning or expression.",
)
def find_orfs(sequence: str, min_aa_length: int = 30, include_reverse: bool = False, **kwargs) -> dict:
    seq, err = _validate_dna(sequence)
    if err:
        return {"summary": err, "error": "invalid_sequence"}

    min_aa = max(5, int(min_aa_length))
    scans = [("forward", seq)]
    if include_reverse:
        scans.append(("reverse", _reverse_complement(seq)))

    orfs = []
    for strand, dna in scans:
        for frame in (0, 1, 2):
            i = frame
            while i <= len(dna) - 3:
                codon = dna[i : i + 3]
                if codon != "ATG":
                    i += 3
                    continue
                j = i + 3
                while j <= len(dna) - 3:
                    stop = dna[j : j + 3]
                    if stop in _STOP_CODONS:
                        aa_len = (j + 3 - i) // 3
                        if aa_len >= min_aa:
                            nt_seq = dna[i : j + 3]
                            orfs.append(
                                {
                                    "strand": strand,
                                    "frame": frame + 1,
                                    "start": i + 1,
                                    "end": j + 3,
                                    "length_nt": len(nt_seq),
                                    "length_aa": aa_len,
                                    "protein": _translate_dna(nt_seq, frame=1, to_stop=True),
                                }
                            )
                        break
                    j += 3
                i += 3

    orfs.sort(key=lambda x: x["length_aa"], reverse=True)
    return {
        "summary": f"Found {len(orfs)} ORFs with length >= {min_aa} aa.",
        "orfs": orfs,
        "count": len(orfs),
    }


@registry.register(
    name="dna.codon_optimize",
    description="Codon-optimize a protein sequence for a host species",
    category="dna",
    parameters={
        "protein_sequence": "Amino-acid sequence (single-letter, may include *)",
        "species": "Target host codon table: human or ecoli",
    },
    usage_guide="Use for expression construct design in common hosts.",
)
def codon_optimize(protein_sequence: str, species: str = "human", **kwargs) -> dict:
    protein = re.sub(r"\s+", "", str(protein_sequence or "").upper())
    protein = protein.replace("*", "")
    if not protein:
        return {"summary": "protein_sequence is required.", "error": "missing_protein"}

    host = str(species or "human").strip().lower()
    if host not in _PREF_CODONS:
        return {"summary": "Unsupported species. Use human or ecoli.", "error": "invalid_species"}

    mapping = _PREF_CODONS[host]
    invalid = sorted({aa for aa in protein if aa not in mapping})
    if invalid:
        return {"summary": f"Invalid amino acids: {', '.join(invalid)}", "error": "invalid_protein"}

    dna = "".join(mapping[aa] for aa in protein)
    return {
        "summary": f"Codon-optimized sequence generated for {host} ({len(protein)} aa).",
        "species": host,
        "protein_length": len(protein),
        "optimized_dna": dna,
        "gc_content": round(_gc_content(dna), 2),
    }


@registry.register(
    name="dna.restriction_sites",
    description="Find common restriction enzyme sites in a DNA sequence",
    category="dna",
    parameters={
        "sequence": "DNA sequence",
        "enzymes": "Optional list or comma-separated enzyme names",
    },
    usage_guide="Use to choose cloning strategy and verify unwanted cut sites.",
)
def restriction_sites(sequence: str, enzymes: list[str] | str | None = None, **kwargs) -> dict:
    seq, err = _validate_dna(sequence)
    if err:
        return {"summary": err, "error": "invalid_sequence"}

    if enzymes is None:
        selected = list(_ENZYME_MOTIFS.keys())
    elif isinstance(enzymes, str):
        selected = [x.strip() for x in enzymes.split(",") if x.strip()]
    else:
        selected = [str(x).strip() for x in enzymes if str(x).strip()]

    unknown = [e for e in selected if e not in _ENZYME_MOTIFS]
    if unknown:
        return {
            "summary": f"Unknown enzymes: {', '.join(unknown)}",
            "error": "invalid_enzyme",
            "available_enzymes": sorted(_ENZYME_MOTIFS.keys()),
        }

    matches = []
    for enzyme in selected:
        motif = _ENZYME_MOTIFS[enzyme]
        positions = _find_sites(seq, motif)
        matches.append(
            {
                "enzyme": enzyme,
                "motif": motif,
                "n_sites": len(positions),
                "positions": positions,
            }
        )

    total = sum(m["n_sites"] for m in matches)
    return {
        "summary": f"Restriction scan complete: {total} total sites across {len(selected)} enzymes.",
        "sequence_length": len(seq),
        "results": matches,
    }


@registry.register(
    name="dna.virtual_digest",
    description="Perform an in-silico digest and return fragment sizes",
    category="dna",
    parameters={
        "sequence": "DNA sequence",
        "enzymes": "List or comma-separated enzymes (supported: EcoRI,BamHI,HindIII,NotI,XhoI,NheI,BsaI,BsmBI)",
        "circular": "Treat sequence as circular (default false)",
    },
    usage_guide="Use to predict gel band patterns before running wet-lab digests.",
)
def virtual_digest(sequence: str, enzymes: list[str] | str, circular: bool = False, **kwargs) -> dict:
    seq, err = _validate_dna(sequence)
    if err:
        return {"summary": err, "error": "invalid_sequence"}

    site_result = restriction_sites(seq, enzymes=enzymes)
    if site_result.get("error"):
        return site_result

    cut_positions = sorted({p for item in site_result["results"] for p in item["positions"]})
    if not cut_positions:
        return {
            "summary": "No cut sites found.",
            "fragments_bp": [len(seq)],
            "n_fragments": 1,
            "cut_positions": [],
        }

    cuts = [p - 1 for p in cut_positions]  # 0-based
    if circular:
        cuts = sorted(cuts)
        fragments = []
        for idx, cut in enumerate(cuts):
            nxt = cuts[(idx + 1) % len(cuts)]
            if nxt > cut:
                fragments.append(nxt - cut)
            else:
                fragments.append((len(seq) - cut) + nxt)
    else:
        points = [0] + cuts + [len(seq)]
        fragments = [points[i + 1] - points[i] for i in range(len(points) - 1) if points[i + 1] - points[i] > 0]

    fragments = sorted(fragments, reverse=True)
    return {
        "summary": f"Virtual digest produced {len(fragments)} fragments.",
        "n_fragments": len(fragments),
        "fragments_bp": fragments,
        "cut_positions": cut_positions,
        "circular": bool(circular),
    }


@registry.register(
    name="dna.primer_design",
    description="Design simple PCR primers around a target region",
    category="dna",
    parameters={
        "sequence": "Template DNA sequence",
        "target_start": "Target region start (1-based, optional)",
        "target_end": "Target region end (1-based, optional)",
        "primer_min_len": "Minimum primer length (default 18)",
        "primer_max_len": "Maximum primer length (default 24)",
        "tm_target": "Target primer Tm in C (default 60)",
    },
    usage_guide="Use as a fast first-pass primer suggestion before detailed wet-lab validation.",
)
def primer_design(
    sequence: str,
    target_start: int | None = None,
    target_end: int | None = None,
    primer_min_len: int = 18,
    primer_max_len: int = 24,
    tm_target: float = 60.0,
    **kwargs,
) -> dict:
    seq, err = _validate_dna(sequence)
    if err:
        return {"summary": err, "error": "invalid_sequence"}

    min_len = max(16, int(primer_min_len))
    max_len = max(min_len, min(35, int(primer_max_len)))
    tm_target = float(tm_target)

    start = int(target_start) if target_start else 1
    end = int(target_end) if target_end else len(seq)
    start = max(1, min(start, len(seq)))
    end = max(start, min(end, len(seq)))

    left_window = seq[max(0, start - 1 - 80) : start - 1 + max_len]
    right_window = seq[max(0, end - max_len) : min(len(seq), end + 80)]

    left_cands = _pick_primer_candidates(left_window, min_len, max_len, tm_target)
    right_rev = _reverse_complement(right_window)
    right_cands = _pick_primer_candidates(right_rev, min_len, max_len, tm_target)

    if not left_cands or not right_cands:
        return {"summary": "Unable to design primers in target windows.", "error": "design_failed"}

    fwd = left_cands[0]
    rev = right_cands[0]

    # Approximate amplicon based on target bounds (not exact genomic placement).
    amplicon_bp = max(1, end - start + 1 + len(fwd.seq) + len(rev.seq))

    return {
        "summary": f"Designed primer pair (F {len(fwd.seq)} nt, R {len(rev.seq)} nt) for ~{amplicon_bp} bp amplicon.",
        "forward_primer": {
            "sequence": fwd.seq,
            "length": len(fwd.seq),
            "tm_c": round(fwd.tm, 2),
            "gc_percent": round(fwd.gc, 2),
        },
        "reverse_primer": {
            "sequence": rev.seq,
            "length": len(rev.seq),
            "tm_c": round(rev.tm, 2),
            "gc_percent": round(rev.gc, 2),
        },
        "target_region": {"start": start, "end": end},
        "estimated_amplicon_bp": amplicon_bp,
        "note": "Heuristic first-pass design; verify specificity with BLAST/in-silico PCR.",
    }


@registry.register(
    name="dna.pcr_protocol",
    description="Generate a PCR thermal cycling protocol",
    category="dna",
    parameters={
        "product_size_bp": "Expected amplicon size",
        "primer_tm": "Primer melting temperature in C",
        "polymerase": "Polymerase name (default Q5)",
        "cycles": "Number of PCR cycles (default 30)",
    },
    usage_guide="Use to quickly draft PCR conditions aligned with primer and amplicon properties.",
)
def pcr_protocol(
    product_size_bp: int = 1000,
    primer_tm: float = 60.0,
    polymerase: str = "Q5",
    cycles: int = 30,
    **kwargs,
) -> dict:
    size_bp = max(50, int(product_size_bp))
    tm = float(primer_tm)
    cycles = max(15, min(40, int(cycles)))

    anneal = max(45.0, min(72.0, tm - 3.0))
    extension_s = max(10, int(round(size_bp / 1000.0 * 30)))

    protocol = [
        {"step": "Initial denaturation", "temperature_c": 98, "time_s": 30},
        {"step": f"{cycles} cycles: denaturation", "temperature_c": 98, "time_s": 10},
        {"step": f"{cycles} cycles: annealing", "temperature_c": round(anneal, 1), "time_s": 20},
        {"step": f"{cycles} cycles: extension", "temperature_c": 72, "time_s": extension_s},
        {"step": "Final extension", "temperature_c": 72, "time_s": 120},
        {"step": "Hold", "temperature_c": 4, "time_s": 0},
    ]

    return {
        "summary": f"PCR protocol generated for {size_bp} bp product using {polymerase}.",
        "polymerase": polymerase,
        "cycles": cycles,
        "product_size_bp": size_bp,
        "annealing_temp_c": round(anneal, 1),
        "extension_time_s": extension_s,
        "protocol": protocol,
        "note": "General starting conditions; optimize empirically for template/primers/polymerase buffer.",
    }


@registry.register(
    name="dna.gibson_design",
    description="Suggest overlap sequences for Gibson assembly fragments",
    category="dna",
    parameters={
        "fragments": "Ordered list of DNA fragment sequences",
        "overlap_length": "Desired overlap length (default 25)",
    },
    usage_guide="Use to draft overlap strategy for multi-fragment Gibson assembly.",
)
def gibson_design(fragments: list[str], overlap_length: int = 25, **kwargs) -> dict:
    if not isinstance(fragments, list) or len(fragments) < 2:
        return {"summary": "Provide at least two fragment sequences.", "error": "invalid_fragments"}

    ov = max(15, min(60, int(overlap_length)))
    cleaned = []
    for idx, fragment in enumerate(fragments, 1):
        seq, err = _validate_dna(fragment)
        if err:
            return {"summary": f"Fragment {idx}: {err}", "error": "invalid_sequence"}
        if len(seq) < ov:
            return {"summary": f"Fragment {idx} shorter than overlap_length {ov}.", "error": "fragment_too_short"}
        cleaned.append(seq)

    joins = []
    for i in range(len(cleaned) - 1):
        left = cleaned[i][-ov:]
        right = cleaned[i + 1][:ov]
        joins.append(
            {
                "join": f"{i + 1}->{i + 2}",
                "left_tail": left,
                "right_head": right,
                "gc_percent": round((_gc_content(left) + _gc_content(right)) / 2.0, 2),
            }
        )

    return {
        "summary": f"Generated Gibson overlap plan for {len(cleaned)} fragments ({len(joins)} joins).",
        "overlap_length": ov,
        "joins": joins,
        "note": "Ensure overlaps are unique and avoid strong secondary structures.",
    }


@registry.register(
    name="dna.golden_gate_design",
    description="Suggest Golden Gate-compatible overhang plan",
    category="dna",
    parameters={
        "parts": "Ordered list of DNA part names or labels",
        "enzyme": "Type IIS enzyme (BsaI or BsmBI; default BsaI)",
    },
    usage_guide="Use to draft overhang strategy for modular Golden Gate assemblies.",
)
def golden_gate_design(parts: list[str], enzyme: str = "BsaI", **kwargs) -> dict:
    if not isinstance(parts, list) or len(parts) < 2:
        return {"summary": "Provide at least two part labels.", "error": "invalid_parts"}

    enzyme_norm = str(enzyme or "BsaI").strip()
    if enzyme_norm not in {"BsaI", "BsmBI"}:
        return {"summary": "Unsupported enzyme. Use BsaI or BsmBI.", "error": "invalid_enzyme"}

    # Simple deterministic non-palindromic overhang set.
    overhang_pool = ["AATG", "GCTT", "CGAA", "TGCC", "ACTA", "GGCT", "TTAC", "CAGG", "AGTC", "TCGA"]
    n_joins = len(parts) - 1
    if n_joins > len(overhang_pool):
        return {"summary": "Too many parts for built-in overhang pool.", "error": "too_many_parts"}

    joins = []
    for i in range(n_joins):
        joins.append(
            {
                "from_part": str(parts[i]),
                "to_part": str(parts[i + 1]),
                "overhang": overhang_pool[i],
            }
        )

    motif = _ENZYME_MOTIFS[enzyme_norm]
    return {
        "summary": f"Golden Gate plan generated for {len(parts)} parts using {enzyme_norm}.",
        "enzyme": enzyme_norm,
        "enzyme_motif": motif,
        "joins": joins,
        "note": "Validate overhang uniqueness and absence of internal Type IIS sites before synthesis.",
    }
