"""
Experimental assay design tools: protocol templates, timeline estimation, and assay listing.

Provides structured experimental protocols for TPD/molecular glue validation workflows,
covering degradation, binding, viability, expression, structural, and screening assays.
"""

import math
from ct.tools import registry


# ---------------------------------------------------------------------------
# Assay protocol templates (TPD / molecular-glue relevant)
# ---------------------------------------------------------------------------

ASSAY_TEMPLATES: dict[str, dict] = {
    "hibit": {
        "name": "HiBiT Degradation Assay",
        "description": "Promega HiBiT lytic/kinetic assay measuring target protein degradation via split-NanoLuc luminescence",
        "category": "degradation",
        "protocol_steps": [
            "Plate HiBiT-tagged cells in 384-well white plates (2000 cells/well)",
            "Allow cells to adhere overnight (16-20 h)",
            "Prepare compound serial dilutions in DMSO (typically 10-point, 3-fold)",
            "Add compounds to plates using acoustic dispenser or pin tool",
            "Incubate at 37C / 5% CO2 for desired timepoints (2h, 6h, 24h)",
            "Add Nano-Glo HiBiT Lytic Reagent (1:1 v/v)",
            "Incubate 10 min at room temperature on orbital shaker",
            "Read luminescence on plate reader (integration time 0.5s)",
            "Normalize to DMSO controls and calculate DC50/Dmax",
        ],
        "reagents": [
            "HiBiT-tagged cell line (endogenous CRISPR knock-in preferred)",
            "Nano-Glo HiBiT Lytic Detection System (Promega N3030)",
            "384-well white plates (Corning 3570)",
            "DMSO (cell-culture grade)",
            "Complete growth medium",
        ],
        "controls": {
            "positive": "Known degrader of target (or proteasome inhibitor MG132 as rescue)",
            "negative": "DMSO vehicle control",
        },
        "readout": "Luminescence (RLU) proportional to target protein level",
        "hands_on_hours": 3.0,
        "calendar_days": 3,
        "cost_per_plate": 800,
    },
    "nanobret": {
        "name": "NanoBRET Ternary Complex Assay",
        "description": "Measures proximity between E3 ligase and target protein via BRET energy transfer in live cells",
        "category": "binding",
        "protocol_steps": [
            "Co-transfect HEK293 cells with NanoLuc-E3 and HaloTag-target constructs",
            "Plate transfected cells into 384-well plates (8000 cells/well)",
            "Allow 24 h for expression",
            "Add HaloTag NanoBRET 618 Ligand (200 nM final)",
            "Add test compounds in dose-response",
            "Incubate 2-4 h at 37C",
            "Add NanoBRET Nano-Glo Substrate",
            "Read donor (460 nm) and acceptor (618 nm) emissions",
            "Calculate milliBRET ratio: (618/460) x 1000",
        ],
        "reagents": [
            "NanoLuc-E3 ligase fusion construct",
            "HaloTag-target fusion construct",
            "HaloTag NanoBRET 618 Ligand (Promega G9801)",
            "NanoBRET Nano-Glo Substrate (Promega N1571)",
            "Transfection reagent (FuGENE HD or Lipofectamine 3000)",
        ],
        "controls": {
            "positive": "Known molecular glue or PROTAC forming ternary complex",
            "negative": "DMSO + no-acceptor control (donor-only)",
        },
        "readout": "milliBRET ratio (higher = stronger ternary complex)",
        "hands_on_hours": 5.0,
        "calendar_days": 4,
        "cost_per_plate": 1200,
    },
    "western_blot": {
        "name": "Western Blot Degradation Confirmation",
        "description": "Orthogonal confirmation of target protein degradation by immunoblotting",
        "category": "degradation",
        "protocol_steps": [
            "Seed cells in 6-well plates (500K cells/well)",
            "Treat with compound at multiple doses and timepoints",
            "Include MG132 co-treatment to confirm proteasome dependence",
            "Lyse cells in RIPA buffer with protease/phosphatase inhibitors",
            "Quantify protein by BCA assay and normalize loading (20-30 ug)",
            "Run SDS-PAGE (4-12% Bis-Tris gel)",
            "Transfer to PVDF membrane (wet transfer, 100V 1h or semi-dry)",
            "Block in 5% BSA/TBST 1h at RT",
            "Probe with primary antibody overnight at 4C",
            "Wash, secondary HRP antibody 1h at RT",
            "Develop with ECL substrate and image",
            "Quantify band intensity and normalize to loading control (vinculin/GAPDH)",
        ],
        "reagents": [
            "Primary antibody against target protein",
            "Anti-vinculin or anti-GAPDH loading control antibody",
            "HRP-conjugated secondary antibodies",
            "RIPA lysis buffer",
            "Protease inhibitor cocktail",
            "4-12% Bis-Tris SDS-PAGE gels (NuPAGE)",
            "PVDF membrane",
            "ECL substrate (SuperSignal West Pico/Femto)",
        ],
        "controls": {
            "positive": "MG132 rescue (5 uM, 2h pre-treatment) to confirm UPS dependence",
            "negative": "DMSO vehicle and untreated cells",
        },
        "readout": "Band intensity (densitometry) normalized to loading control",
        "hands_on_hours": 8.0,
        "calendar_days": 3,
        "cost_per_plate": 200,
    },
    "qpcr": {
        "name": "qPCR Transcript-Level Analysis",
        "description": "RT-qPCR to confirm degradation is post-transcriptional (mRNA unchanged) or detect transcriptional effects",
        "category": "expression",
        "protocol_steps": [
            "Treat cells with compound (same conditions as degradation assay)",
            "Extract total RNA using RNeasy or TRIzol",
            "Quantify RNA (NanoDrop) and check quality (A260/280 > 1.8)",
            "Reverse-transcribe 1 ug RNA with oligo-dT primers",
            "Design qPCR primers spanning exon-exon junctions",
            "Set up qPCR reactions in 384-well plates (10 uL volume)",
            "Run qPCR: 95C 10min, 40x(95C 15s, 60C 1min), melt curve",
            "Analyze by delta-delta-Ct method normalized to housekeeping genes",
        ],
        "reagents": [
            "RNA extraction kit (Qiagen RNeasy or TRIzol)",
            "Reverse transcription kit (SuperScript IV or High-Capacity cDNA)",
            "SYBR Green or TaqMan master mix",
            "Target-specific primers (2 sets for redundancy)",
            "Housekeeping gene primers (GAPDH, ACTB, HPRT1)",
            "384-well qPCR plates",
        ],
        "controls": {
            "positive": "Known transcriptional modulator (e.g., actinomycin D for mRNA stability)",
            "negative": "DMSO vehicle; no-RT control for genomic DNA contamination",
        },
        "readout": "Relative mRNA expression (fold change vs DMSO)",
        "hands_on_hours": 6.0,
        "calendar_days": 2,
        "cost_per_plate": 300,
    },
    "ctg_viability": {
        "name": "CellTiter-Glo Cell Viability",
        "description": "ATP-based luminescent cell viability assay for dose-response curves and IC50 determination",
        "category": "viability",
        "protocol_steps": [
            "Plate cells in 384-well white plates (500-2000 cells/well depending on growth rate)",
            "Allow overnight adherence",
            "Add compounds in dose-response (10-point, 3-fold dilution, 10 uM top dose)",
            "Incubate 72 h at 37C / 5% CO2",
            "Equilibrate plates to room temperature (30 min)",
            "Add CellTiter-Glo reagent (1:1 v/v)",
            "Shake 2 min, incubate 10 min at RT",
            "Read luminescence",
            "Fit 4-parameter logistic curve to calculate IC50",
        ],
        "reagents": [
            "CellTiter-Glo 2.0 (Promega G9241)",
            "384-well white plates",
            "Complete growth medium",
            "DMSO (cell-culture grade)",
        ],
        "controls": {
            "positive": "Staurosporine (1 uM) or bortezomib as cytotoxic control",
            "negative": "DMSO vehicle control",
        },
        "readout": "Luminescence (RLU) proportional to ATP/viable cells; IC50 from curve fit",
        "hands_on_hours": 2.0,
        "calendar_days": 4,
        "cost_per_plate": 500,
    },
    "flow_cytometry": {
        "name": "Flow Cytometry",
        "description": "Multi-parameter flow cytometry for surface marker expression, cell death (Annexin V/PI), and cell cycle analysis",
        "category": "viability",
        "protocol_steps": [
            "Treat cells with compound at chosen doses/timepoints",
            "Harvest cells (trypsinize adherent or collect suspension)",
            "Wash 2x with cold PBS",
            "For surface markers: stain with fluorochrome-conjugated antibodies (30 min, 4C, dark)",
            "For apoptosis: stain with Annexin V-FITC and PI per kit protocol",
            "For cell cycle: fix in 70% ethanol, stain with PI/RNase A",
            "Acquire on flow cytometer (minimum 10,000 events/sample)",
            "Analyze with FlowJo or similar software",
            "Gate on singlets, then live cells, then markers of interest",
        ],
        "reagents": [
            "Fluorochrome-conjugated antibodies for markers of interest",
            "Annexin V-FITC/PI Apoptosis Kit",
            "Propidium iodide + RNase A (for cell cycle)",
            "FACS buffer (PBS + 2% FBS + 0.1% NaN3)",
            "70% ethanol (cell-cycle fixation)",
            "CompBeads for compensation",
        ],
        "controls": {
            "positive": "Staurosporine (apoptosis) or nocodazole (G2/M arrest)",
            "negative": "DMSO vehicle; unstained and single-stain compensation controls",
        },
        "readout": "Percentage of cells in each population (live, apoptotic, necrotic; cell cycle phase)",
        "hands_on_hours": 6.0,
        "calendar_days": 2,
        "cost_per_plate": 400,
    },
    "tr_fret": {
        "name": "TR-FRET Binding Assay",
        "description": "Time-resolved FRET assay for measuring binary binding (compound-protein) or ternary complex formation",
        "category": "binding",
        "protocol_steps": [
            "Prepare assay buffer (50 mM HEPES pH 7.5, 150 mM NaCl, 0.01% Tween-20, 0.1% BSA)",
            "Dispense DMSO/compound into 384-well low-volume plates",
            "Add Eu-labeled donor protein (e.g., Eu-anti-His for His-tagged E3)",
            "Add AF647-labeled acceptor protein (e.g., AF647-target)",
            "For ternary complex: add both proteins + compound simultaneously",
            "Incubate 1-2 h at room temperature",
            "Read TR-FRET: excite 320 nm, read 665/620 nm ratio",
            "Calculate FRET ratio and fit dose-response",
        ],
        "reagents": [
            "Europium-labeled donor (anti-tag antibody or direct protein label)",
            "AlexaFluor647-labeled acceptor protein",
            "Purified recombinant proteins (E3 ligase, target)",
            "384-well low-volume black plates (Corning 4514)",
            "Assay buffer components",
        ],
        "controls": {
            "positive": "Known binder at saturating concentration",
            "negative": "DMSO vehicle; protein-only (no compound) for baseline FRET",
        },
        "readout": "FRET ratio (665 nm / 620 nm); EC50 from dose-response",
        "hands_on_hours": 4.0,
        "calendar_days": 1,
        "cost_per_plate": 600,
    },
    "alphalisa": {
        "name": "AlphaLISA Protein-Protein Interaction",
        "description": "Bead-based proximity assay for detecting protein-protein interactions and ternary complex formation",
        "category": "binding",
        "protocol_steps": [
            "Prepare assay buffer (25 mM HEPES pH 7.4, 100 mM NaCl, 0.1% BSA, 0.01% Tween-20)",
            "Dispense compounds into 384-well AlphaPlates",
            "Add biotinylated protein 1 and His-tagged protein 2",
            "Incubate 1 h at room temperature",
            "Add Anti-His AlphaLISA Acceptor beads (10 ug/mL final)",
            "Incubate 1 h at room temperature",
            "Add Streptavidin Donor beads (40 ug/mL final) in subdued light",
            "Incubate 1 h at room temperature in dark",
            "Read on Alpha-compatible reader (EnVision or CLARIOstar)",
        ],
        "reagents": [
            "Biotinylated protein 1",
            "His-tagged protein 2",
            "Anti-His AlphaLISA Acceptor beads (PerkinElmer AL128)",
            "Streptavidin Alpha Donor beads (PerkinElmer 6760002)",
            "384-well AlphaPlates (PerkinElmer 6005350)",
            "Assay buffer components",
        ],
        "controls": {
            "positive": "Known PPI stabilizer or molecular glue at saturating concentration",
            "negative": "DMSO vehicle; beads-only (no protein) for background",
        },
        "readout": "Alpha signal (counts); EC50 from dose-response curve",
        "hands_on_hours": 4.0,
        "calendar_days": 1,
        "cost_per_plate": 900,
    },
    "dsf": {
        "name": "Differential Scanning Fluorimetry (Thermal Shift)",
        "description": "Measures compound-induced thermal stabilization of target protein as evidence of direct binding",
        "category": "structural",
        "protocol_steps": [
            "Prepare protein at 2-5 uM in assay buffer (PBS or HEPES-based, low detergent)",
            "Add SYPRO Orange dye (5x final concentration)",
            "Dispense 18 uL protein/dye mix into 384-well PCR plates",
            "Add 2 uL compound (10x stock) or DMSO control",
            "Seal plates with optical adhesive film",
            "Run thermal ramp on qPCR instrument: 25C to 95C at 1C/min",
            "Monitor SYPRO Orange fluorescence (Ex 470, Em 570)",
            "Determine Tm by fitting Boltzmann sigmoid to melt curves",
            "Calculate delta-Tm = Tm(compound) - Tm(DMSO)",
        ],
        "reagents": [
            "Purified recombinant target protein (>90% purity)",
            "SYPRO Orange Protein Gel Stain (Invitrogen S6650)",
            "384-well PCR plates (optically clear)",
            "Optical adhesive film",
            "Assay buffer (PBS or 50 mM HEPES, 150 mM NaCl, pH 7.5)",
        ],
        "controls": {
            "positive": "Known ligand that shifts Tm by >2C",
            "negative": "DMSO vehicle (matched % DMSO); protein-only for intrinsic Tm",
        },
        "readout": "Melting temperature (Tm) shift in degrees C; delta-Tm > 2C suggests binding",
        "hands_on_hours": 3.0,
        "calendar_days": 1,
        "cost_per_plate": 150,
    },
    "spr": {
        "name": "Surface Plasmon Resonance (SPR)",
        "description": "Label-free real-time binding kinetics: measures ka, kd, and KD for compound-protein interactions",
        "category": "structural",
        "protocol_steps": [
            "Immobilize target protein on CM5 sensor chip via amine coupling",
            "Inject EDC/NHS to activate surface, then protein (10-50 ug/mL in acetate pH 4.0-5.5)",
            "Block remaining sites with ethanolamine",
            "Prepare compound dilution series in running buffer (+ matched DMSO%)",
            "Inject compound at multiple concentrations (single-cycle or multi-cycle kinetics)",
            "Monitor association (120-180 s) and dissociation (300-600 s)",
            "Regenerate surface between cycles if needed (10 mM glycine pH 2.0)",
            "Fit sensorgrams to 1:1 Langmuir model to extract ka, kd, KD",
            "Perform solvent correction for DMSO bulk effects",
        ],
        "reagents": [
            "CM5 sensor chip (Cytiva BR100530)",
            "Amine Coupling Kit (Cytiva BR100050)",
            "Purified target protein (>95% purity, activity confirmed)",
            "Running buffer (HBS-EP+: 10 mM HEPES, 150 mM NaCl, 3 mM EDTA, 0.05% P20)",
            "Regeneration buffer (10 mM glycine-HCl pH 2.0)",
        ],
        "controls": {
            "positive": "Known binder with published KD for assay validation",
            "negative": "Reference channel (no protein) for bulk refractive index subtraction",
        },
        "readout": "Binding kinetics: ka (1/Ms), kd (1/s), KD (M); steady-state affinity if kinetics too fast",
        "hands_on_hours": 8.0,
        "calendar_days": 2,
        "cost_per_plate": 2000,
    },
    "tmt_proteomics": {
        "name": "TMT Multiplexed Proteomics",
        "description": "Global proteomics by TMT labeling and LC-MS/MS to profile degradation selectivity across the proteome",
        "category": "screening",
        "protocol_steps": [
            "Treat cells (1-5 million per condition) with compound vs DMSO (3+ replicates)",
            "Lyse in 8M urea / 50 mM TEAB buffer",
            "Reduce (TCEP 10 mM, 30 min 37C) and alkylate (IAA 20 mM, 30 min RT dark)",
            "Digest with trypsin (1:50 enzyme:protein, overnight 37C)",
            "Desalt on C18 Sep-Pak cartridges",
            "Label peptides with TMT reagents (TMT-16plex or TMT-18plex)",
            "Combine labeled samples, desalt, and fractionate by high-pH reversed-phase (8-12 fractions)",
            "Analyze each fraction by nanoLC-MS/MS (2h gradient, Orbitrap or timsTOF)",
            "Search with MaxQuant or Proteome Discoverer",
            "Filter: 1% FDR at peptide and protein level",
            "Statistical analysis: limma or MSstats for differential abundance",
        ],
        "reagents": [
            "TMT-16plex or TMT-18plex reagent kit (Thermo A44520 / A52045)",
            "Trypsin (sequencing grade, Promega V5111)",
            "TCEP and iodoacetamide",
            "C18 Sep-Pak cartridges (Waters WAT054955)",
            "High-pH reversed-phase fractionation kit",
            "nanoLC-MS/MS instrument access (Orbitrap Exploris/Eclipse or timsTOF)",
        ],
        "controls": {
            "positive": "Include known degrader condition as positive control channel",
            "negative": "DMSO vehicle (3+ replicates for statistical power)",
        },
        "readout": "Log2 fold-change per protein (compound vs DMSO); volcano plot of selectivity",
        "hands_on_hours": 20.0,
        "calendar_days": 14,
        "cost_per_plate": 5000,
    },
    "crispr_screen": {
        "name": "Genome-Wide CRISPR Screen",
        "description": "Pooled CRISPR knockout screen to identify genetic dependencies and resistance/sensitization mechanisms",
        "category": "screening",
        "protocol_steps": [
            "Expand cells to sufficient scale (500-1000x library coverage, e.g., 100M cells for 100K library)",
            "Transduce with lentiviral sgRNA library at MOI 0.3",
            "Select with puromycin (2-4 ug/mL) for 48-72 h",
            "Confirm >30% transduction by flow cytometry (if GFP reporter)",
            "Split into treatment arms: compound vs DMSO (maintain 500x coverage)",
            "Treat for 14-21 days (passage every 3 days, re-dose compound)",
            "Harvest cells, extract genomic DNA",
            "PCR-amplify sgRNA cassettes with indexed primers",
            "Sequence on NextSeq/NovaSeq (aim for 500 reads per sgRNA)",
            "Analyze with MAGeCK or CRISPR-CASA to identify depleted/enriched genes",
        ],
        "reagents": [
            "Lentiviral sgRNA library (Brunello, TKOv3, or custom focused library)",
            "Puromycin selection antibiotic",
            "Genomic DNA extraction kit (Qiagen Blood & Cell Culture DNA Midi)",
            "PCR primers for sgRNA amplification (library-specific)",
            "NextSeq/NovaSeq sequencing access",
            "MAGeCK software for analysis",
        ],
        "controls": {
            "positive": "Essential gene sgRNAs should deplete (e.g., core essential gene list from Hart et al.)",
            "negative": "Non-targeting sgRNAs and DMSO-treated arm for baseline",
        },
        "readout": "Gene-level enrichment/depletion scores; beta scores from MAGeCK-MLE",
        "hands_on_hours": 40.0,
        "calendar_days": 35,
        "cost_per_plate": 15000,
    },
}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@registry.register(
    name="experiment.list_assays",
    description="List all available experimental assay templates with name, description, and category",
    category="experiment",
    parameters={},
    usage_guide="You want to see all available experimental assay types before designing a specific protocol.",
)
def list_assays(**kwargs) -> dict:
    """Return a catalogue of all available assay templates."""
    assays = []
    for key, tmpl in ASSAY_TEMPLATES.items():
        assays.append({
            "assay_type": key,
            "name": tmpl["name"],
            "description": tmpl["description"],
            "category": tmpl["category"],
        })

    by_category: dict[str, list[str]] = {}
    for a in assays:
        by_category.setdefault(a["category"], []).append(a["assay_type"])

    cat_summary = "; ".join(f"{cat}: {', '.join(types)}" for cat, types in sorted(by_category.items()))

    return {
        "summary": f"{len(assays)} assay templates available across {len(by_category)} categories. {cat_summary}",
        "assays": assays,
        "categories": by_category,
    }


@registry.register(
    name="experiment.design_assay",
    description="Design a detailed experimental protocol for a specific assay type, customized with target/compound/cell line information",
    category="experiment",
    parameters={
        "assay_type": "Assay template key (e.g. 'hibit', 'nanobret', 'tmt_proteomics')",
        "target": "Target protein or gene name",
        "compound": "Compound name or identifier",
        "cell_line": "Cell line to use",
        "goal": "Specific experimental goal or question",
    },
    usage_guide="You need to design an experimental protocol for validating a computational finding. Use before experiment.estimate_timeline and cro.match_experiment.",
)
def design_assay(
    assay_type: str,
    target: str = None,
    compound: str = None,
    cell_line: str = None,
    goal: str = None,
    **kwargs,
) -> dict:
    """Design a customized experimental protocol from a template."""
    if assay_type not in ASSAY_TEMPLATES:
        available = ", ".join(sorted(ASSAY_TEMPLATES.keys()))
        return {"error": f"Unknown assay type '{assay_type}'. Available: {available}", "summary": f"Unknown assay type '{assay_type}'. Available: {available}"}
    tmpl = ASSAY_TEMPLATES[assay_type]

    # Customize protocol steps with target/compound/cell_line info
    customized_steps = []
    for step in tmpl["protocol_steps"]:
        s = step
        if target:
            s = s.replace("target protein", f"{target} protein")
            s = s.replace("target gene", f"{target} gene")
        if compound:
            s = s.replace("compound", compound).replace("test compounds", compound)
        if cell_line:
            s = s.replace("cells", f"{cell_line} cells").replace("Plate cells", f"Plate {cell_line} cells")
        customized_steps.append(s)

    # Build customized controls
    controls = dict(tmpl["controls"])
    if target:
        controls["positive"] = controls["positive"].replace("target", target)
    if compound:
        controls["negative"] = controls["negative"].replace("compound", compound)

    # Build context string
    context_parts = []
    if target:
        context_parts.append(f"target={target}")
    if compound:
        context_parts.append(f"compound={compound}")
    if cell_line:
        context_parts.append(f"cell_line={cell_line}")
    if goal:
        context_parts.append(f"goal={goal}")
    context_str = ", ".join(context_parts) if context_parts else "generic protocol"

    # Assemble protocol
    protocol = {
        "assay_type": assay_type,
        "name": tmpl["name"],
        "description": tmpl["description"],
        "category": tmpl["category"],
        "context": context_str,
        "protocol_steps": customized_steps,
        "reagents": list(tmpl["reagents"]),
        "controls": controls,
        "readout": tmpl["readout"],
        "estimated_hands_on_hours": tmpl["hands_on_hours"],
        "estimated_calendar_days": tmpl["calendar_days"],
        "estimated_cost_per_plate": tmpl["cost_per_plate"],
    }

    if goal:
        protocol["experimental_goal"] = goal

    summary_parts = [f"Designed {tmpl['name']} protocol"]
    if target:
        summary_parts.append(f"for {target}")
    if compound:
        summary_parts.append(f"with {compound}")
    if cell_line:
        summary_parts.append(f"in {cell_line}")
    summary_parts.append(
        f"({len(customized_steps)} steps, ~{tmpl['hands_on_hours']}h hands-on, "
        f"{tmpl['calendar_days']} calendar days, ~${tmpl['cost_per_plate']}/plate)"
    )

    return {
        "summary": ". ".join(summary_parts),
        "protocol": protocol,
    }


@registry.register(
    name="experiment.estimate_timeline",
    description="Estimate hands-on time, calendar time, and cost for an experiment scaled by number of compounds, replicates, and doses",
    category="experiment",
    parameters={
        "assay_type": "Assay template key (e.g. 'hibit', 'ctg_viability')",
        "n_compounds": "Number of compounds to test",
        "n_replicates": "Number of biological replicates",
        "n_doses": "Number of dose points per compound",
    },
    usage_guide="You need to estimate how long an experiment will take and how much it will cost. Use after experiment.design_assay.",
)
def estimate_timeline(
    assay_type: str,
    n_compounds: int = 1,
    n_replicates: int = 3,
    n_doses: int = 8,
    **kwargs,
) -> dict:
    """Estimate timeline and cost scaled by experimental parameters."""
    if assay_type not in ASSAY_TEMPLATES:
        available = ", ".join(sorted(ASSAY_TEMPLATES.keys()))
        return {"error": f"Unknown assay type '{assay_type}'. Available: {available}", "summary": f"Unknown assay type '{assay_type}'. Available: {available}"}
    tmpl = ASSAY_TEMPLATES[assay_type]

    # Calculate number of wells needed
    # Each compound x dose x replicate = 1 well, plus controls (~10% overhead)
    wells_per_compound = n_doses * n_replicates
    total_wells = n_compounds * wells_per_compound
    control_wells = max(16, int(total_wells * 0.1))  # at least 16 control wells
    total_wells_with_controls = total_wells + control_wells

    # Number of 384-well plates needed
    wells_per_plate = 384
    n_plates = math.ceil(total_wells_with_controls / wells_per_plate)

    # Scale hands-on time: base time per plate, with efficiency gains for batching
    base_hours = tmpl["hands_on_hours"]
    # First plate takes full time, each additional plate adds ~60% of base
    hands_on_hours = base_hours + max(0, n_plates - 1) * base_hours * 0.6
    hands_on_days = round(hands_on_hours / 8.0, 1)

    # Calendar days: base + extra days for additional plates (batching helps)
    base_cal_days = tmpl["calendar_days"]
    extra_plate_days = max(0, math.ceil((n_plates - 1) / 4))  # ~4 plates per batch
    calendar_days = base_cal_days + extra_plate_days

    # Cost
    total_cost = n_plates * tmpl["cost_per_plate"]

    return {
        "summary": (
            f"{tmpl['name']}: {n_compounds} compounds x {n_doses} doses x {n_replicates} replicates = "
            f"{total_wells_with_controls} wells ({n_plates} plates). "
            f"Estimated {hands_on_hours:.1f}h hands-on ({hands_on_days} days), "
            f"{calendar_days} calendar days, ~${total_cost:,}"
        ),
        "assay_type": assay_type,
        "assay_name": tmpl["name"],
        "n_compounds": n_compounds,
        "n_doses": n_doses,
        "n_replicates": n_replicates,
        "total_wells": total_wells_with_controls,
        "n_plates": n_plates,
        "hands_on_hours": round(hands_on_hours, 1),
        "hands_on_days": hands_on_days,
        "calendar_days": calendar_days,
        "estimated_cost": total_cost,
        "cost_per_plate": tmpl["cost_per_plate"],
    }
