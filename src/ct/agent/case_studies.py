"""Curated case study definitions and runner for ct.

Provides pre-defined multi-agent research profiles for landmark drugs,
designed for demos, conference presentations, and sales collateral.

Each case study defines 3-4 complementary research angles that are
executed in parallel via the ResearchOrchestrator (bypassing the LLM
decomposition step).
"""

import logging
from dataclasses import dataclass, field

from ct.agent.orchestrator import OrchestratorResult, ResearchOrchestrator, ThreadGoal

logger = logging.getLogger("ct.case_studies")


@dataclass
class CaseStudy:
    """A curated drug case study for demo/showcase purposes."""

    id: str
    name: str
    compound: str
    targets: list[str]
    indication: str
    description: str
    thread_goals: list[dict] = field(default_factory=list)


# ─── Case study registry ──────────────────────────────────────

CASE_STUDIES: dict[str, CaseStudy] = {
    "revlimid": CaseStudy(
        id="revlimid",
        name="Revlimid (lenalidomide)",
        compound="lenalidomide",
        targets=["CRBN", "IKZF1", "IKZF3"],
        indication="multiple myeloma",
        description=(
            "Lenalidomide is the poster child of molecular glue degraders — "
            "an IMiD that hijacks CRBN to degrade IKZF1/3 transcription factors. "
            "A $12B/yr blockbuster that transformed myeloma treatment."
        ),
        thread_goals=[
            {
                "angle": "Target Biology & Mechanism",
                "goal": (
                    "Investigate CRBN E3 ligase biology, IKZF1/IKZF3 neosubstrate "
                    "degradation mechanism, and co-essential gene networks for "
                    "lenalidomide targets in multiple myeloma."
                ),
                "suggested_tools": [
                    "target.druggability",
                    "target.expression_profile",
                    "target.coessentiality",
                    "target.disease_association",
                ],
            },
            {
                "angle": "Chemical Analysis & SAR",
                "goal": (
                    "Analyze lenalidomide structure, SAR landscape, find similar "
                    "IMiD compounds in ChEMBL, and assess ADMET properties."
                ),
                "suggested_tools": [
                    "chemistry.similarity_search",
                    "chemistry.sar_analyze",
                    "chemistry.descriptors",
                    "safety.admet_predict",
                ],
            },
            {
                "angle": "Clinical Landscape",
                "goal": (
                    "Map lenalidomide clinical indications, competitive landscape "
                    "in multiple myeloma, active clinical trials, and patient "
                    "population estimates."
                ),
                "suggested_tools": [
                    "clinical.indication_map",
                    "clinical.competitive_landscape",
                    "clinical.trial_search",
                    "clinical.population_size",
                ],
            },
            {
                "angle": "Safety & Resistance",
                "goal": (
                    "Profile lenalidomide safety: SALL4 risk, antitarget activity, "
                    "known resistance mutations in CRBN, and biomarker strategy "
                    "for patient selection."
                ),
                "suggested_tools": [
                    "safety.sall4_risk",
                    "safety.antitarget_profile",
                    "biomarker.resistance_profile",
                    "biomarker.mutation_sensitivity",
                ],
            },
        ],
    ),
    "gleevec": CaseStudy(
        id="gleevec",
        name="Gleevec (imatinib)",
        compound="imatinib",
        targets=["BCR-ABL", "ABL1", "KIT", "PDGFRA"],
        indication="chronic myeloid leukemia",
        description=(
            "Imatinib revolutionized cancer therapy as the first rationally "
            "designed kinase inhibitor. It transformed CML from a fatal "
            "diagnosis to a manageable chronic condition."
        ),
        thread_goals=[
            {
                "angle": "Target Biology & Selectivity",
                "goal": (
                    "Investigate BCR-ABL fusion biology, ABL1 kinase domain, "
                    "off-target kinase activity on KIT and PDGFRA, and "
                    "expression profiles across leukemia subtypes."
                ),
                "suggested_tools": [
                    "target.druggability",
                    "target.expression_profile",
                    "data_api.uniprot_lookup",
                    "network.ppi_analysis",
                ],
            },
            {
                "angle": "Structural & Chemical Analysis",
                "goal": (
                    "Analyze imatinib binding mode, SAR of 2-phenylaminopyrimidine "
                    "scaffold, compare with nilotinib/dasatinib, and assess "
                    "key pharmacophore features."
                ),
                "suggested_tools": [
                    "chemistry.sar_analyze",
                    "chemistry.similarity_search",
                    "chemistry.pharmacophore",
                    "structure.binding_site",
                ],
            },
            {
                "angle": "Resistance & Biomarkers",
                "goal": (
                    "Profile imatinib resistance mutations (T315I gatekeeper), "
                    "resistance mechanisms, and biomarker-guided treatment "
                    "selection strategies in CML."
                ),
                "suggested_tools": [
                    "biomarker.resistance_profile",
                    "biomarker.mutation_sensitivity",
                    "genomics.variant_annotate",
                    "literature.pubmed_search",
                ],
            },
        ],
    ),
    "keytruda": CaseStudy(
        id="keytruda",
        name="Keytruda (pembrolizumab)",
        compound="pembrolizumab",
        targets=["PD-1", "PDCD1"],
        indication="non-small cell lung cancer",
        description=(
            "Pembrolizumab is the world's best-selling drug ($25B/yr) — "
            "a PD-1 checkpoint inhibitor that unleashes the immune system "
            "against cancer. Approved in 30+ tumor types."
        ),
        thread_goals=[
            {
                "angle": "Immuno-Oncology Mechanism",
                "goal": (
                    "Investigate PD-1/PD-L1 checkpoint biology, immune evasion "
                    "mechanisms, and immune cell infiltration patterns across "
                    "pembrolizumab-responsive tumor types."
                ),
                "suggested_tools": [
                    "target.expression_profile",
                    "target.disease_association",
                    "expression.immune_score",
                    "expression.deconvolution",
                ],
            },
            {
                "angle": "Biomarker Strategy",
                "goal": (
                    "Analyze pembrolizumab biomarkers: PD-L1 expression (TPS/CPS), "
                    "MSI-H/dMMR, TMB, and emerging biomarkers for response "
                    "prediction across indications."
                ),
                "suggested_tools": [
                    "biomarker.panel_select",
                    "biomarker.mutation_sensitivity",
                    "genomics.gwas_lookup",
                    "literature.pubmed_search",
                ],
            },
            {
                "angle": "Clinical Positioning",
                "goal": (
                    "Map pembrolizumab across 30+ approved indications, "
                    "competitive landscape vs nivolumab/atezolizumab, ongoing "
                    "combination trials, and market positioning."
                ),
                "suggested_tools": [
                    "clinical.indication_map",
                    "clinical.competitive_landscape",
                    "clinical.trial_search",
                    "clinical.population_size",
                ],
            },
            {
                "angle": "Combination & Resistance",
                "goal": (
                    "Explore pembrolizumab combination strategies (chemo, TKIs, "
                    "other I/O agents), mechanisms of acquired resistance, and "
                    "potential synergy partners."
                ),
                "suggested_tools": [
                    "combination.synergy_predict",
                    "biomarker.resistance_profile",
                    "literature.pubmed_search",
                    "expression.pathway_enrichment",
                ],
            },
        ],
    ),
    "ozempic": CaseStudy(
        id="ozempic",
        name="Ozempic (semaglutide)",
        compound="semaglutide",
        targets=["GLP1R"],
        indication="type 2 diabetes / obesity",
        description=(
            "Semaglutide is the GLP-1 receptor agonist behind the obesity "
            "revolution. Originally for T2D, it crossed over into obesity, "
            "cardiovascular protection, and potentially NASH/MASH."
        ),
        thread_goals=[
            {
                "angle": "Target Biology & Signaling",
                "goal": (
                    "Investigate GLP-1 receptor biology, downstream signaling "
                    "cascades, expression across metabolic tissues, and "
                    "structural basis for semaglutide binding."
                ),
                "suggested_tools": [
                    "target.druggability",
                    "target.expression_profile",
                    "data_api.uniprot_lookup",
                    "expression.pathway_enrichment",
                ],
            },
            {
                "angle": "Clinical Indications & Expansion",
                "goal": (
                    "Map semaglutide clinical indications (T2D, obesity, CVOT), "
                    "pipeline expansion into NASH/MASH and Alzheimer's, "
                    "competitive landscape vs tirzepatide."
                ),
                "suggested_tools": [
                    "clinical.indication_map",
                    "clinical.competitive_landscape",
                    "clinical.trial_search",
                    "clinical.population_size",
                ],
            },
            {
                "angle": "Safety & Pharmacology",
                "goal": (
                    "Assess semaglutide safety profile: GI side effects, "
                    "thyroid C-cell risk, pancreatitis signal, and long-term "
                    "cardiovascular outcomes data."
                ),
                "suggested_tools": [
                    "safety.admet_predict",
                    "safety.classify",
                    "literature.pubmed_search",
                    "data_api.drug_info",
                ],
            },
        ],
    ),
    "xalkori": CaseStudy(
        id="xalkori",
        name="Xalkori (crizotinib)",
        compound="crizotinib",
        targets=["ALK", "ROS1", "MET"],
        indication="ALK+ non-small cell lung cancer",
        description=(
            "Crizotinib pioneered precision oncology — the first ALK inhibitor "
            "approved alongside a companion diagnostic. It proved that matching "
            "drugs to molecular subtypes transforms outcomes."
        ),
        thread_goals=[
            {
                "angle": "Target Biology & Fusions",
                "goal": (
                    "Investigate ALK fusion biology (EML4-ALK), ROS1 "
                    "rearrangements, MET amplification, and expression/dependency "
                    "profiles across NSCLC subtypes."
                ),
                "suggested_tools": [
                    "target.druggability",
                    "target.expression_profile",
                    "target.disease_association",
                    "genomics.variant_annotate",
                ],
            },
            {
                "angle": "Chemical SAR & Next-Gen Inhibitors",
                "goal": (
                    "Analyze crizotinib SAR (aminopyridine scaffold), compare "
                    "with next-gen ALK inhibitors (alectinib, lorlatinib), "
                    "and structural basis for selectivity."
                ),
                "suggested_tools": [
                    "chemistry.sar_analyze",
                    "chemistry.similarity_search",
                    "chemistry.descriptors",
                    "structure.binding_site",
                ],
            },
            {
                "angle": "Resistance & Companion Diagnostics",
                "goal": (
                    "Profile crizotinib resistance mutations (L1196M, G1269A), "
                    "next-gen ALK inhibitor strategies, and companion diagnostic "
                    "requirements for ALK/ROS1 testing."
                ),
                "suggested_tools": [
                    "biomarker.resistance_profile",
                    "biomarker.mutation_sensitivity",
                    "biomarker.panel_select",
                    "literature.pubmed_search",
                ],
            },
        ],
    ),
}


def build_thread_goals(case: CaseStudy) -> list[ThreadGoal]:
    """Convert case study thread_goals dicts into ThreadGoal objects.

    Parameters
    ----------
    case : CaseStudy
        The case study whose thread goals to build.

    Returns
    -------
    list[ThreadGoal]
        Ordered list of ThreadGoal objects with sequential IDs.
    """
    goals = []
    for i, g in enumerate(case.thread_goals, start=1):
        goals.append(
            ThreadGoal(
                thread_id=i,
                angle=g["angle"],
                goal=g["goal"],
                suggested_tools=g.get("suggested_tools", []),
            )
        )
    return goals


def run_case_study(
    session,
    case_id: str,
    n_threads: int = None,
) -> OrchestratorResult:
    """Run a curated case study through the multi-agent orchestrator.

    Parameters
    ----------
    session : Session
        Active ct session.
    case_id : str
        Case study slug (e.g. "revlimid", "gleevec").
    n_threads : int, optional
        Number of threads. Defaults to the number of thread goals defined
        in the case study.

    Returns
    -------
    OrchestratorResult
        Merged results from all research threads.

    Raises
    ------
    ValueError
        If *case_id* is not found in the registry.
    """
    if case_id not in CASE_STUDIES:
        available = ", ".join(sorted(CASE_STUDIES.keys()))
        raise ValueError(
            f"Unknown case study '{case_id}'. Available: {available}"
        )

    case = CASE_STUDIES[case_id]
    goals = build_thread_goals(case)

    if n_threads is None:
        n_threads = len(goals)

    # Build query from case study metadata
    query = (
        f"Comprehensive analysis of {case.name}: {case.description} "
        f"Compound: {case.compound}. "
        f"Primary targets: {', '.join(case.targets)}. "
        f"Indication: {case.indication}."
    )

    context = {
        "compound": case.compound,
        "targets": ", ".join(case.targets),
        "indication": case.indication,
        "case_study": case.id,
    }

    orchestrator = ResearchOrchestrator(session, n_threads=n_threads)
    return orchestrator.run(query, context, preset_goals=goals)
