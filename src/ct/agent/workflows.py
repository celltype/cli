"""
Workflow templates for common drug discovery research patterns.

These are injected into the planner prompt to guide tool selection and sequencing.
The planner can follow, adapt, or combine workflows as needed.
"""


WORKFLOWS = {
    "target_validation": {
        "description": "Validate a potential drug target",
        "trigger_phrases": [
            "validate target", "is this a good target", "target assessment",
            "druggable target", "target validation",
        ],
        "steps": [
            {"tool": "target.coessentiality", "why": "Find functionally related genes and synthetic lethal partners"},
            {"tool": "literature.pubmed_search", "why": "Check published validation data and known biology"},
            {"tool": "clinical.indication_map", "why": "Map to cancer indications with PRISM sensitivity data"},
            {"tool": "biomarker.mutation_sensitivity", "why": "Check if mutations in the target affect drug response"},
            {"tool": "clinical.tcga_stratify", "why": "Assess target expression across cancer types via TCGA"},
        ],
    },
    "compound_safety": {
        "description": "Full safety assessment of a compound",
        "trigger_phrases": [
            "safety assessment", "is this compound safe", "toxicity profile",
            "safety check", "off-target",
        ],
        "steps": [
            {"tool": "safety.antitarget_profile", "why": "Screen for off-target degradation of tumor suppressors and essential proteins"},
            {"tool": "safety.sall4_risk", "why": "Assess teratogenicity risk via SALL4 degradation (IMiD-type liability)"},
            {"tool": "safety.classify", "why": "Get overall SAFE/CAUTION/DANGEROUS verdict combining all signals"},
            {"tool": "viability.tissue_selectivity", "why": "Check for broad vs selective killing across tissue types"},
        ],
    },
    "hit_characterization": {
        "description": "Characterize a hit compound from a screen",
        "trigger_phrases": [
            "characterize compound", "hit characterization", "what does this compound do",
            "compound profile", "hit profiling",
        ],
        "steps": [
            {"tool": "chemistry.descriptors", "why": "Get molecular properties, drug-likeness, and Lipinski profile"},
            {"tool": "viability.dose_response", "why": "Understand potency across cell lines with IC50 estimates"},
            {"tool": "expression.pathway_enrichment", "why": "Identify affected pathways from L1000 transcriptomic signature"},
            {"tool": "safety.classify", "why": "Quick safety classification before advancing the compound"},
            {"tool": "literature.chembl_query", "why": "Find related compounds and known bioactivity in ChEMBL"},
        ],
    },
    "combination_therapy": {
        "description": "Design a combination therapy strategy",
        "trigger_phrases": [
            "combination therapy", "synergy", "combine with", "drug combination",
            "synthetic lethality", "combination strategy",
        ],
        "steps": [
            {"tool": "combination.synergy_predict", "why": "Find synergistic partners via anti-correlated transcriptomic signatures"},
            {"tool": "combination.synthetic_lethality", "why": "Mine DepMap for synthetic lethal gene pairs"},
            {"tool": "combination.metabolic_vulnerability", "why": "Identify exploitable metabolic dependencies for combination"},
            {"tool": "expression.immune_score", "why": "Check IO potential for immuno-oncology combinations"},
        ],
    },
    "clinical_positioning": {
        "description": "Position a compound for clinical development",
        "trigger_phrases": [
            "clinical positioning", "which indication", "patient population",
            "go-to-market", "clinical strategy", "indication selection",
        ],
        "steps": [
            {"tool": "clinical.indication_map", "why": "Map compound sensitivity to cancer indications"},
            {"tool": "clinical.population_size", "why": "Estimate addressable patient populations per indication"},
            {"tool": "biomarker.mutation_sensitivity", "why": "Identify predictive biomarkers for patient selection"},
            {"tool": "clinical.tcga_stratify", "why": "Validate target expression in patient tumors via TCGA"},
            {"tool": "clinical.trial_design_benchmark", "why": "Benchmark endpoint and protocol design patterns in the current trial landscape"},
            {"tool": "literature.pubmed_search", "why": "Review clinical landscape and competitor data"},
        ],
    },
    "cro_engagement": {
        "description": "Design experiment and engage a CRO for outsourced work",
        "trigger_phrases": [
            "find a CRO", "outsource experiment", "CRO inquiry",
            "contract research", "send to CRO",
        ],
        "steps": [
            {"tool": "experiment.design_assay", "why": "Generate a detailed assay protocol"},
            {"tool": "experiment.estimate_timeline", "why": "Get time and cost estimates for the experiment"},
            {"tool": "cro.match_experiment", "why": "Find best-fit CROs for the assay type"},
            {"tool": "cro.draft_inquiry", "why": "Generate a professional inquiry email to the top CRO"},
        ],
    },
    "structure_prediction": {
        "description": "Predict ternary complex structure for a molecular glue",
        "trigger_phrases": [
            "predict structure", "ternary complex", "structural prediction",
            "dock compound", "binding mode",
        ],
        "steps": [
            {"tool": "structure.alphafold_fetch", "why": "Download AlphaFold structure for the target protein"},
            {"tool": "structure.compound_3d", "why": "Generate 3D conformer for the compound"},
            {"tool": "structure.ternary_predict", "why": "Predict ternary complex (E3 + compound + target)"},
            {"tool": "chemistry.descriptors", "why": "Get molecular properties for structure-activity context"},
        ],
    },
    "gpu_computation": {
        "description": "Submit and manage GPU compute jobs",
        "trigger_phrases": [
            "GPU computation", "run Boltz", "run AlphaFold", "submit job",
            "cloud compute", "estimate cost",
        ],
        "steps": [
            {"tool": "compute.estimate_cost", "why": "Get cost and time estimate before committing resources"},
            {"tool": "compute.list_providers", "why": "Review available GPU providers and pricing"},
            {"tool": "compute.submit_job", "why": "Submit the computation job (dry_run by default)"},
        ],
    },
    "custom_analysis": {
        "description": "Custom data exploration or visualization",
        "trigger_phrases": [
            "create a plot", "make a visualization", "custom analysis",
            "heatmap", "volcano plot", "statistical test", "scatter plot",
        ],
        "steps": [
            {"tool": "code.execute", "why": "Generate custom analysis code"},
        ],
    },
    "script_authoring": {
        "description": "Write or update a standalone script/code file in the workspace",
        "trigger_phrases": [
            "write a python script", "save as .py", "create script file",
            "generate a script",
        ],
        "steps": [
            {"tool": "files.create_file", "why": "Create the requested script file with full code content"},
            {"tool": "files.read_file", "why": "Verify file contents after writing"},
        ],
    },
    "report_generation": {
        "description": "Generate and save a research report",
        "trigger_phrases": [
            "write a report", "save report", "export findings",
            "generate report", "create a report",
        ],
        "steps": [
            {"tool": "code.execute", "why": "Run analysis and gather data"},
            {"tool": "files.write_report", "why": "Save formatted report to output directory"},
        ],
    },
    "genetic_evidence": {
        "description": "Build a comprehensive genetic evidence case for a target-disease link",
        "trigger_phrases": [
            "genetic evidence", "causal evidence", "Mendelian randomization",
            "GWAS evidence", "genetic validation", "causal link",
        ],
        "steps": [
            {"tool": "genomics.gwas_lookup", "why": "Find genome-wide significant associations"},
            {"tool": "genomics.eqtl_lookup", "why": "Check expression QTLs across tissues"},
            {"tool": "genomics.mendelian_randomization_lookup", "why": "Assess causal evidence via MR"},
            {"tool": "genomics.coloc", "why": "Test GWAS-eQTL colocalization (shared causal variant)"},
            {"tool": "target.expression_profile", "why": "Understand tissue expression pattern"},
        ],
    },
    "lead_optimization": {
        "description": "Optimize a hit compound into a lead",
        "trigger_phrases": [
            "optimize compound", "lead optimization", "improve potency",
            "SAR", "improve ADMET", "make analogs",
        ],
        "steps": [
            {"tool": "chemistry.sar_analyze", "why": "Understand current SAR landscape"},
            {"tool": "chemistry.mmp_analysis", "why": "Find matched molecular pair transformations that improve properties"},
            {"tool": "chemistry.scaffold_hop", "why": "Generate scaffold-hopped analogs for IP space"},
            {"tool": "design.suggest_modifications", "why": "Get medicinal chemistry modification suggestions"},
            {"tool": "safety.admet_predict", "why": "Predict ADMET for top candidates"},
            {"tool": "chemistry.retrosynthesis", "why": "Check synthetic accessibility of top analogs"},
        ],
    },
    "protein_deep_dive": {
        "description": "Comprehensive protein characterization",
        "trigger_phrases": [
            "tell me about this protein", "protein function", "protein structure",
            "domain architecture", "protein characterization",
        ],
        "steps": [
            {"tool": "protein.function_predict", "why": "Get full UniProt annotation: function, location, GO terms"},
            {"tool": "protein.domain_annotate", "why": "Map domain architecture from InterPro"},
            {"tool": "data_api.pdb_search", "why": "Find experimental structures"},
            {"tool": "target.expression_profile", "why": "Tissue expression from GTEx and HPA"},
            {"tool": "network.ppi_analysis", "why": "Map protein interaction partners"},
        ],
    },
    "drug_repurposing": {
        "description": "Find repurposing opportunities for existing drugs",
        "trigger_phrases": [
            "repurpose", "drug repurposing", "new indication",
            "repositioning", "off-label", "existing drug for",
        ],
        "steps": [
            {"tool": "repurposing.cmap_query", "why": "Match drug expression signature to disease signatures"},
            {"tool": "data_api.drug_info", "why": "Get comprehensive drug profile and known indications"},
            {"tool": "clinical.trial_search", "why": "Check ongoing trials in the new indication"},
            {"tool": "literature.patent_search", "why": "Assess IP landscape for new indication"},
            {"tool": "clinical.competitive_landscape", "why": "Map competitors in the target indication"},
        ],
    },
    "molecular_docking": {
        "description": "Dock compounds into a protein target and analyze binding",
        "trigger_phrases": [
            "dock", "docking", "binding mode", "binding site",
            "virtual screening", "binding affinity",
        ],
        "steps": [
            {"tool": "structure.alphafold_fetch", "why": "Get protein structure (AlphaFold if no experimental)"},
            {"tool": "structure.binding_site", "why": "Identify druggable binding pockets"},
            {"tool": "structure.compound_3d", "why": "Generate 3D ligand conformer"},
            {"tool": "structure.dock", "why": "Dock ligand into binding site"},
            {"tool": "design.suggest_modifications", "why": "Suggest modifications to improve binding"},
        ],
    },
    "resistance_analysis": {
        "description": "Analyze drug resistance mechanisms and predict resistance-associated biomarkers",
        "trigger_phrases": [
            "resistance mechanism", "resistance profile", "drug resistance",
            "resistance mutation", "acquired resistance", "resistance biomarker",
        ],
        "steps": [
            {"tool": "biomarker.mutation_sensitivity", "why": "Identify mutations that alter drug sensitivity — potential resistance drivers"},
            {"tool": "expression.l1000_similarity", "why": "Find compounds with similar transcriptomic signatures to identify resistance-associated expression patterns"},
            {"tool": "expression.pathway_enrichment", "why": "Map resistance-associated expression changes to pathways (e.g., efflux, bypass signaling)"},
            {"tool": "literature.pubmed_search", "why": "Search published literature for known resistance mechanisms to this drug/target class"},
            {"tool": "biomarker.resistance_profile", "why": "Build comprehensive resistance profile combining mutation, expression, and literature data"},
        ],
    },
    "therapeutic_window": {
        "description": "Assess therapeutic window by comparing on-target vs off-target toxicity",
        "trigger_phrases": [
            "therapeutic window", "therapeutic index", "selectivity index",
            "on-target toxicity", "off-target toxicity", "safety margin",
        ],
        "steps": [
            {"tool": "viability.dose_response", "why": "Get dose-response in target cancer cell lines to establish efficacy range"},
            {"tool": "viability.tissue_selectivity", "why": "Compare sensitivity across tissue types to identify selective vs broadly toxic profiles"},
            {"tool": "viability.tissue_selectivity", "why": "Compare sensitivity across lineages to calculate therapeutic window between sensitive and resistant tissue types"},
            {"tool": "safety.antitarget_profile", "why": "Screen for off-target degradation of tumor suppressors and essential proteins"},
            {"tool": "safety.classify", "why": "Get overall safety classification combining all toxicity signals"},
        ],
    },
    "competitive_landscape": {
        "description": "Map the competitive landscape for a drug target or indication",
        "trigger_phrases": [
            "competitive landscape", "competitor analysis", "market landscape",
            "clinical pipeline", "what drugs target", "who else is developing",
        ],
        "steps": [
            {"tool": "clinical.competitive_landscape", "why": "Aggregate competitive intelligence from Open Targets, ChEMBL, and ClinicalTrials.gov"},
            {"tool": "clinical.trial_search", "why": "Search ClinicalTrials.gov for active and recruiting trials in the indication"},
            {"tool": "literature.pubmed_search", "why": "Find recent publications on clinical results and competitor compounds"},
            {"tool": "clinical.indication_map", "why": "Map compound sensitivity to cancer indications to identify positioning opportunities"},
        ],
    },
    "treatment_landscape": {
        "description": "Describe standard of care treatment and where a drug class fits",
        "trigger_phrases": [
            "standard of care", "treatment sequencing", "treatment regimen",
            "approved therapies", "treatment landscape", "where do",
        ],
        "steps": [
            {"tool": "literature.pubmed_search", "why": "Search for current treatment guidelines and landmark trials"},
            {"tool": "clinical.trial_search", "why": "Search ClinicalTrials.gov for current and recent trials establishing standard of care"},
            {"tool": "clinical.competitive_landscape", "why": "Map all approved and investigational drugs for the indication"},
            {"tool": "data_api.opentargets_search", "why": "Get Open Targets disease-level drug and target landscape"},
        ],
    },
    "mutation_resistance": {
        "description": "Identify clinically observed resistance mutations for a drug or drug class",
        "trigger_phrases": [
            "resistance mutation", "clinically observed mutation", "mutation frequency",
            "IMiD resistance", "drug resistance mutation", "acquired resistance",
        ],
        "steps": [
            {"tool": "literature.pubmed_search", "why": "Search for publications on clinically observed resistance mutations"},
            {"tool": "data_api.opentargets_search", "why": "Get known genetic associations and somatic mutations from Open Targets"},
            {"tool": "biomarker.mutation_sensitivity", "why": "Check if mutations correlate with drug sensitivity in preclinical data"},
            {"tool": "biomarker.resistance_profile", "why": "Build comprehensive resistance profile"},
            {"tool": "literature.openalex_search", "why": "Search for additional clinical mutation data in recent literature"},
        ],
    },
    "protac_design": {
        "description": "Analyze PROTAC linker and component properties",
        "trigger_phrases": [
            "PROTAC", "linker length", "linker composition", "bifunctional degrader",
            "dBET", "MZ1", "ARV", "PROTAC design",
        ],
        "steps": [
            {"tool": "chemistry.pubchem_lookup", "why": "Look up PROTAC structures and molecular properties"},
            {"tool": "chemistry.descriptors", "why": "Calculate molecular descriptors including MW, logP, TPSA for PROTACs"},
            {"tool": "literature.chembl_query", "why": "Find ChEMBL bioactivity data for PROTACs"},
            {"tool": "literature.pubmed_search", "why": "Search for PROTAC SAR and linkerology publications"},
        ],
    },
    "compound_comparison": {
        "description": "Compare two or more compounds on activity and selectivity",
        "trigger_phrases": [
            "compare compounds", "versus", "differential sensitivity",
            "compare selectivity", "compare potency", "which is more potent",
        ],
        "steps": [
            {"tool": "chemistry.pubchem_lookup", "why": "Look up each compound separately to get structures and properties"},
            {"tool": "viability.dose_response", "why": "Get dose-response for first compound (run separately for each)"},
            {"tool": "viability.tissue_selectivity", "why": "Get tissue selectivity for first compound"},
            {"tool": "literature.pubmed_search", "why": "Search published head-to-head comparisons"},
            {"tool": "literature.chembl_query", "why": "Get bioactivity data from ChEMBL for comparison"},
        ],
    },
    "patient_population": {
        "description": "Estimate addressable patient population for a drug concept",
        "trigger_phrases": [
            "patient population", "addressable population", "market sizing",
            "how many patients", "incidence", "prevalence",
        ],
        "steps": [
            {"tool": "clinical.population_size", "why": "Get SEER incidence data for the indication"},
            {"tool": "clinical.trial_search", "why": "Check current trials for treatment rates and eligible populations"},
            {"tool": "clinical.competitive_landscape", "why": "Understand competitive landscape and unmet need"},
            {"tool": "literature.pubmed_search", "why": "Find epidemiology data and treatment utilization rates"},
        ],
    },
    "omics_scrnaseq_analysis": {
        "description": "Analyze single-cell RNA-seq data for a gene or disease",
        "trigger_phrases": [
            "single-cell analysis", "scRNA-seq", "analyze single-cell",
            "single cell RNA", "cell type composition",
        ],
        "steps": [
            {"tool": "omics.geo_search", "why": "Search GEO for relevant scRNA-seq datasets"},
            {"tool": "omics.cellxgene_search", "why": "Search CELLxGENE for curated single-cell datasets"},
            {"tool": "omics.geo_fetch", "why": "Download the most relevant dataset"},
            {"tool": "omics.dataset_info", "why": "Inspect dataset structure and metadata"},
            {"tool": "singlecell.cluster", "why": "Cluster cells and identify populations"},
            {"tool": "singlecell.cell_type_annotate", "why": "Annotate cell types using marker genes"},
            {"tool": "expression.pathway_enrichment", "why": "Identify enriched pathways in cell populations"},
        ],
    },
    "omics_bulk_analysis": {
        "description": "Analyze bulk RNA-seq or expression data",
        "trigger_phrases": [
            "bulk RNA-seq", "bulk expression", "differential expression from GEO",
            "analyze expression data", "gene expression dataset",
        ],
        "steps": [
            {"tool": "omics.geo_search", "why": "Search GEO for relevant expression datasets"},
            {"tool": "omics.geo_fetch", "why": "Download the expression matrix"},
            {"tool": "omics.dataset_info", "why": "Inspect dataset structure and dimensions"},
            {"tool": "omics.deseq2", "why": "Run DESeq2 with explicit sample metadata (condition labels) for robust count-based differential expression"},
            {"tool": "expression.pathway_enrichment", "why": "Identify enriched pathways from DEGs"},
        ],
    },
    "omics_data_discovery": {
        "description": "Find and evaluate public datasets for a research question",
        "trigger_phrases": [
            "find dataset", "find public data", "search for datasets",
            "available data", "download data", "data discovery",
        ],
        "steps": [
            {"tool": "omics.geo_search", "why": "Search NCBI GEO for relevant datasets"},
            {"tool": "omics.cellxgene_search", "why": "Search CELLxGENE for curated single-cell data"},
            {"tool": "omics.tcga_search", "why": "Search TCGA/GDC for cancer genomics data"},
            {"tool": "omics.dataset_info", "why": "Inspect and summarize the top dataset"},
        ],
    },
    "omics_methylation_analysis": {
        "description": "Analyze DNA methylation data for differential methylation",
        "trigger_phrases": [
            "methylation analysis", "differential methylation", "DNA methylation",
            "CpG methylation", "epigenetic analysis",
        ],
        "steps": [
            {"tool": "omics.geo_search", "why": "Search GEO for methylation datasets"},
            {"tool": "omics.geo_fetch", "why": "Download methylation beta-value matrix"},
            {"tool": "omics.dataset_info", "why": "Inspect dataset structure"},
            {"tool": "omics.methylation_profile", "why": "Summarize global methylation landscape"},
            {"tool": "omics.methylation_diff", "why": "Identify differentially methylated sites using explicit case/control sample groups"},
            {"tool": "omics.methylation_cluster", "why": "Cluster samples by methylation patterns"},
        ],
    },
    "omics_proteomics_analysis": {
        "description": "Analyze proteomics data for differential protein abundance",
        "trigger_phrases": [
            "proteomics analysis", "differential protein", "protein abundance",
            "mass spectrometry", "TMT proteomics",
        ],
        "steps": [
            {"tool": "omics.dataset_info", "why": "Inspect proteomics data structure"},
            {"tool": "omics.proteomics_diff", "why": "Differential protein abundance analysis with explicit sample grouping"},
            {"tool": "omics.proteomics_enrich", "why": "Pathway enrichment of DE proteins"},
        ],
    },
    "omics_epigenomics_analysis": {
        "description": "Analyze ATAC-seq or ChIP-seq epigenomic data",
        "trigger_phrases": [
            "ATAC-seq analysis", "ChIP-seq analysis", "chromatin accessibility",
            "epigenomic profiling", "open chromatin",
        ],
        "steps": [
            {"tool": "omics.geo_search", "why": "Search GEO for ATAC-seq/ChIP-seq datasets"},
            {"tool": "omics.geo_fetch", "why": "Download peak or count data"},
            {"tool": "omics.atac_peak_annotate", "why": "Annotate peaks by genomic features"},
            {"tool": "omics.chromatin_accessibility", "why": "Differential accessibility between explicit biological groups"},
            {"tool": "omics.chipseq_enrich", "why": "Enrichment analysis of target genes"},
        ],
    },
    "omics_multiomics_integration": {
        "description": "Integrate multiple omics modalities into shared latent space",
        "trigger_phrases": [
            "multi-omics integration", "integrate RNA and ATAC", "MOFA",
            "multiomics", "combine omics modalities",
        ],
        "steps": [
            {"tool": "omics.dataset_info", "why": "Inspect each modality file"},
            {"tool": "omics.multiomics_integrate", "why": "MOFA+ integration into shared latent factors"},
        ],
    },
    "omics_spatial_analysis": {
        "description": "Analyze spatial transcriptomics data",
        "trigger_phrases": [
            "spatial transcriptomics", "Visium", "MERFISH", "spatial gene expression",
            "spatial clustering", "tissue architecture",
        ],
        "steps": [
            {"tool": "omics.cellxgene_search", "why": "Search for spatial datasets"},
            {"tool": "omics.dataset_info", "why": "Inspect spatial data structure"},
            {"tool": "omics.spatial_cluster", "why": "Spatial-aware cell clustering"},
            {"tool": "omics.spatial_autocorrelation", "why": "Identify spatially patterned genes"},
        ],
    },
}


def format_workflows_for_llm(allowed_tools: set[str] | None = None) -> str:
    """Format workflow templates as markdown for the planner prompt."""
    lines = ["\n# Recommended Workflows", ""]
    lines.append(
        "These are expert-recommended tool sequences for common drug discovery tasks. "
        "You may follow, adapt, or combine them as appropriate for the query."
    )
    lines.append("")

    for wf_id, wf in WORKFLOWS.items():
        wf_steps = wf["steps"]
        if allowed_tools is not None:
            wf_steps = [s for s in wf_steps if s["tool"] in allowed_tools]
        if not wf_steps:
            continue

        lines.append(f"## {wf_id}: {wf['description']}")
        triggers = ", ".join(f'"{t}"' for t in wf["trigger_phrases"][:3])
        lines.append(f"  Trigger phrases: {triggers}")
        for i, step in enumerate(wf_steps, 1):
            lines.append(f"  {i}. **{step['tool']}** — {step['why']}")
        lines.append("")

    return "\n".join(lines)
