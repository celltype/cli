"""
Chemistry tools: molecular descriptors, SAR analysis, similarity search.
"""

from ct.tools import registry
from ct.tools.http_client import request


def _extract_smiles(smiles) -> str:
    """Extract a SMILES string from various input types and resolve drug names.

    Handles the case where a dict (e.g., full pubchem_lookup result) is passed
    instead of a plain SMILES string — typically when the planner uses $step.1
    instead of $step.1.canonical_smiles.

    Also resolves drug names (e.g. "lenalidomide") to SMILES via
    _compound_resolver.resolve_to_smiles.
    """
    if isinstance(smiles, dict):
        smiles = (smiles.get("canonical_smiles") or smiles.get("smiles")
                  or smiles.get("summary", ""))
    smiles = str(smiles).strip()

    # Try to resolve name → SMILES (handles both valid SMILES and drug names)
    try:
        from ct.tools._compound_resolver import resolve_to_smiles
        return resolve_to_smiles(smiles)
    except (ValueError, ImportError):
        return smiles  # Fall through — tool will handle invalid SMILES


@registry.register(
    name="chemistry.descriptors",
    description="Compute molecular descriptors and fingerprints for a compound from SMILES",
    category="chemistry",
    parameters={"smiles": "SMILES string"},
    usage_guide="You need molecular properties (MW, LogP, TPSA, Lipinski) for a compound. Use early in hit characterization to assess drug-likeness and physicochemical profile.",
)
def descriptors(smiles: str, **kwargs) -> dict:
    """Compute molecular properties from SMILES."""
    smiles = _extract_smiles(smiles)
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}", "summary": f"Invalid SMILES: {smiles}"}
    props = {
        "smiles": smiles,
        "molecular_weight": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "rings": Descriptors.RingCount(mol),
        "aromatic_rings": Descriptors.NumAromaticRings(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "num_stereocenters": len(Chem.FindMolChiralCenters(mol)),
    }

    # Lipinski Rule of 5
    props["lipinski_violations"] = sum([
        props["molecular_weight"] > 500,
        props["logp"] > 5,
        props["hbd"] > 5,
        props["hba"] > 10,
    ])

    # Molecular glue specific
    props["mw_logp_ratio"] = props["molecular_weight"] / (props["logp"] + 1e-6)
    props["tpsa_per_mw"] = props["tpsa"] / props["molecular_weight"]

    return {
        "summary": f"Molecular profile for {props['formula']} (MW={props['molecular_weight']:.1f}, "
                   f"LogP={props['logp']:.2f}, Lipinski violations={props['lipinski_violations']})",
        "properties": props,
    }


@registry.register(
    name="chemistry.pairwise_similarity",
    description="Compute pairwise Tanimoto similarity matrix for a list of compounds (by name or SMILES)",
    category="chemistry",
    parameters={
        "compounds": "List of compound names or SMILES strings",
        "fingerprint": "Fingerprint type: 'morgan' (default, ECFP4) or 'maccs'",
    },
    usage_guide="You need to compute fingerprint similarity between a set of named compounds. Use when the question asks to 'compare similarity', 'cluster by scaffold', or 'compute Tanimoto' between specific compounds. Returns a full pairwise similarity matrix.",
)
def pairwise_similarity(compounds: list = None, fingerprint: str = "morgan", **kwargs) -> dict:
    """Compute pairwise Tanimoto similarity for a set of compounds."""
    if not compounds or len(compounds) < 2:
        return {"error": "Need at least 2 compounds", "summary": "Provide a list of 2+ compound names or SMILES"}

    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, MACCSkeys

    # Resolve names to SMILES and compute fingerprints
    resolved = []
    for cpd in compounds:
        smi = _extract_smiles(cpd)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            resolved.append({"name": cpd, "smiles": smi, "mol": None, "error": f"Invalid SMILES: {smi}"})
        else:
            resolved.append({"name": cpd, "smiles": Chem.MolToSmiles(mol), "mol": mol})

    # Compute fingerprints
    fps = []
    for r in resolved:
        if r["mol"] is None:
            fps.append(None)
        elif fingerprint == "maccs":
            fps.append(MACCSkeys.GenMACCSKeys(r["mol"]))
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(r["mol"], 2, nBits=2048))

    # Compute pairwise similarity matrix
    n = len(resolved)
    matrix = {}
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if fps[i] is None or fps[j] is None:
                sim = 0.0
            else:
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            key = f"{resolved[i]['name']} vs {resolved[j]['name']}"
            matrix[key] = round(sim, 4)
            pairs.append({
                "compound_a": resolved[i]["name"],
                "compound_b": resolved[j]["name"],
                "smiles_a": resolved[i]["smiles"],
                "smiles_b": resolved[j]["smiles"],
                "tanimoto": round(sim, 4),
            })

    # Sort by similarity
    pairs.sort(key=lambda x: -x["tanimoto"])

    # Cluster suggestion
    if pairs:
        most_similar = pairs[0]
        least_similar = pairs[-1]
    else:
        most_similar = least_similar = {}

    # Build readable matrix
    names = [r["name"] for r in resolved]
    matrix_rows = []
    for i in range(n):
        row = {}
        for j in range(n):
            if i == j:
                row[names[j]] = 1.0
            elif i < j:
                row[names[j]] = round(DataStructs.TanimotoSimilarity(fps[i], fps[j]), 4) if fps[i] and fps[j] else 0.0
            else:
                row[names[j]] = round(DataStructs.TanimotoSimilarity(fps[j], fps[i]), 4) if fps[i] and fps[j] else 0.0
        matrix_rows.append({"compound": names[i], **row})

    fp_label = "ECFP4 (Morgan r=2, 2048 bits)" if fingerprint == "morgan" else "MACCS keys (166 bits)"

    summary_lines = [
        f"Pairwise Tanimoto similarity ({fp_label}) for {n} compounds:",
    ]
    for p in pairs:
        summary_lines.append(f"  {p['compound_a']} vs {p['compound_b']}: {p['tanimoto']:.4f}")
    if most_similar:
        summary_lines.append(f"Most similar: {most_similar['compound_a']} & {most_similar['compound_b']} ({most_similar['tanimoto']:.4f})")
    if least_similar:
        summary_lines.append(f"Most different: {least_similar['compound_a']} & {least_similar['compound_b']} ({least_similar['tanimoto']:.4f})")

    return {
        "summary": "\n".join(summary_lines),
        "fingerprint_type": fp_label,
        "n_compounds": n,
        "pairs": pairs,
        "matrix": matrix_rows,
        "resolved_smiles": [{"name": r["name"], "smiles": r["smiles"]} for r in resolved],
    }


@registry.register(
    name="chemistry.similarity_search",
    description="Find similar compounds in a library using Tanimoto similarity on Morgan fingerprints",
    category="chemistry",
    parameters={"smiles": "Query SMILES", "library_path": "Path to compound library CSV", "top_n": "Number of hits"},
    usage_guide="You have a hit compound and want to find structurally similar analogs in a library. Use for SAR expansion or finding backup compounds with similar scaffolds.",
)
def similarity_search(smiles: str, library_path: str = None, top_n: int = 10, **kwargs) -> dict:
    """Search for similar compounds using fingerprint similarity."""
    smiles = _extract_smiles(smiles)
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    import pandas as pd

    query_mol = Chem.MolFromSmiles(smiles)
    if query_mol is None:
        return {"error": f"Invalid SMILES: {smiles}", "summary": f"Invalid SMILES: {smiles}"}
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)

    # Load library
    if library_path:
        lib = pd.read_csv(library_path)
    else:
        return {"error": "No compound library specified", "summary": "No compound library specified"}
    smiles_col = next((c for c in lib.columns if c.lower() in ['smiles', 'canonical_smiles']), None)
    if smiles_col is None:
        return {"error": f"No SMILES column found in library", "summary": f"No SMILES column found in library"}
    results = []
    for _, row in lib.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        similarity = DataStructs.TanimotoSimilarity(query_fp, fp)
        results.append({
            "smiles": row[smiles_col],
            "similarity": similarity,
            **{k: row[k] for k in row.index if k != smiles_col},
        })

    results.sort(key=lambda x: -x["similarity"])
    top_hits = results[:top_n]

    return {
        "summary": f"Top {top_n} similar compounds (max Tanimoto={top_hits[0]['similarity']:.3f})" if top_hits else "No hits",
        "hits": top_hits,
        "library_size": len(results),
    }


@registry.register(
    name="chemistry.sar_analyze",
    description="Analyze structure-activity relationships for a set of compounds with activity data",
    category="chemistry",
    parameters={"compounds_path": "CSV with SMILES and activity columns"},
    usage_guide="You have a set of compounds with activity data and want to understand which molecular features drive potency. Use for medicinal chemistry optimization guidance.",
)
def sar_analyze(compounds_path: str, activity_col: str = "activity", **kwargs) -> dict:
    """Run SAR analysis on a compound set."""
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    import numpy as np

    df = pd.read_csv(compounds_path)
    smiles_col = next((c for c in df.columns if c.lower() in ['smiles', 'canonical_smiles']), None)

    if smiles_col is None or activity_col not in df.columns:
        return {"error": "Need SMILES and activity columns", "summary": "Need SMILES and activity columns"}
    # Compute descriptors
    features = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol is None:
            continue
        features.append({
            "smiles": row[smiles_col],
            "activity": row[activity_col],
            "mw": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            "rotbonds": Descriptors.NumRotatableBonds(mol),
        })

    feat_df = pd.DataFrame(features)

    # Correlate descriptors with activity
    from scipy import stats
    correlations = {}
    for col in ["mw", "logp", "tpsa", "hbd", "hba", "rotbonds"]:
        r, p = stats.pearsonr(feat_df[col], feat_df["activity"])
        correlations[col] = {"r": round(r, 3), "p": round(p, 4)}

    return {
        "summary": f"SAR analysis on {len(feat_df)} compounds",
        "correlations": correlations,
        "n_compounds": len(feat_df),
    }


@registry.register(
    name="chemistry.mmp_analysis",
    description="Matched molecular pair analysis to identify R-group transformations that improve activity",
    category="chemistry",
    parameters={
        "compounds_csv": "Path to CSV with SMILES and activity columns",
        "activity_col": "Name of the activity column (default 'activity')",
    },
    usage_guide="You have a congeneric series of compounds and want to identify which single-point structural changes drive activity. Use for medicinal chemistry SAR optimization — finds matched molecular pairs and ranks R-group swaps by activity improvement.",
)
def mmp_analysis(compounds_csv: str = None, activity_col: str = "activity", **kwargs) -> dict:
    """Matched molecular pair analysis for a set of compounds.

    Fragments molecules at single acyclic bonds, identifies matched pairs
    (same core, different R-group), and correlates R-group changes with
    activity differences.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import BRICS, AllChem, Descriptors, rdMolDescriptors
    except ImportError:
        return {"error": "RDKit is required for MMP analysis. Install with: pip install rdkit", "summary": "RDKit is required for MMP analysis. Install with: pip install rdkit"}
    import pandas as pd
    import numpy as np

    # Load or generate demo data
    if compounds_csv:
        try:
            df = pd.read_csv(compounds_csv)
        except Exception as e:
            return {"error": f"Could not read CSV: {e}", "summary": f"Failed to load {compounds_csv}"}
        smiles_col = next((c for c in df.columns if c.lower() in ["smiles", "canonical_smiles"]), None)
        if smiles_col is None:
            return {"error": "No SMILES column found (expected 'smiles' or 'canonical_smiles')", "summary": "No SMILES column found (expected 'smiles' or 'canonical_smiles')"}
        if activity_col not in df.columns:
            return {"error": f"Activity column '{activity_col}' not found. Available: {list(df.columns)}", "summary": f"Activity column '{activity_col}' not found. Available: {list(df.columns)}"}
    else:
        # Demo dataset: simple benzamide series
        demo_data = [
            ("c1ccc(C(=O)N)cc1", 5.2, "benzamide"),
            ("c1ccc(C(=O)N)cc1F", 6.1, "4-fluorobenzamide"),
            ("c1ccc(C(=O)N)cc1Cl", 5.8, "4-chlorobenzamide"),
            ("c1ccc(C(=O)N)cc1C", 5.5, "4-methylbenzamide"),
            ("c1ccc(C(=O)N)cc1OC", 6.4, "4-methoxybenzamide"),
            ("c1ccc(C(=O)N)cc1O", 6.0, "4-hydroxybenzamide"),
            ("c1ccc(C(=O)NC)cc1", 5.0, "N-methylbenzamide"),
            ("c1ccc(C(=O)NCC)cc1", 4.7, "N-ethylbenzamide"),
            ("c1ccc(C(=O)N)c(F)c1", 5.9, "3-fluorobenzamide"),
            ("c1cc(F)c(C(=O)N)cc1F", 6.8, "3,4-difluorobenzamide"),
        ]
        df = pd.DataFrame(demo_data, columns=["smiles", activity_col, "name"])
        smiles_col = "smiles"

    # Parse molecules and compute Murcko scaffolds
    from rdkit.Chem.Scaffolds import MurckoScaffold

    parsed = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row[smiles_col])
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smi = Chem.MolToSmiles(scaffold)
        except Exception:
            scaffold_smi = "unknown"
        parsed.append({
            "smiles": row[smiles_col],
            "mol": mol,
            "activity": float(row[activity_col]),
            "scaffold": scaffold_smi,
            "name": row.get("name", row[smiles_col]),
        })

    if len(parsed) < 2:
        return {"error": "Need at least 2 valid compounds for MMP analysis",
                "summary": "Insufficient valid compounds for analysis"}

    # Fragment each molecule using BRICS
    fragments_map = {}  # smiles -> list of (core, rgroup) tuples
    for entry in parsed:
        mol = entry["mol"]
        smi = entry["smiles"]
        fragments_map[smi] = []

        try:
            brics_frags = BRICS.BRICSDecompose(mol, returnMols=False)
            for frag in brics_frags:
                fragments_map[smi].append(frag)
        except Exception:
            pass

    # Identify matched pairs: same scaffold, different compounds
    scaffold_groups = {}
    for entry in parsed:
        scaffold_groups.setdefault(entry["scaffold"], []).append(entry)

    pairs = []
    transformations = {}  # (from_feature, to_feature) -> [delta_activity]

    for scaffold, members in scaffold_groups.items():
        if len(members) < 2:
            continue

        # Generate all pairs within scaffold group
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                m1 = members[i]
                m2 = members[j]
                delta = m2["activity"] - m1["activity"]

                # Find structural difference using MCS
                try:
                    from rdkit.Chem import rdFMCS
                    mcs = rdFMCS.FindMCS(
                        [m1["mol"], m2["mol"]],
                        timeout=2,
                        matchValences=True,
                        ringMatchesRingOnly=True,
                    )
                    core_smarts = mcs.smartsString if mcs and mcs.numAtoms > 0 else None
                except Exception:
                    core_smarts = None

                # Characterize the transformation by atom count difference
                atoms1 = m1["mol"].GetNumHeavyAtoms()
                atoms2 = m2["mol"].GetNumHeavyAtoms()

                pair_info = {
                    "compound_a": m1["smiles"],
                    "compound_b": m2["smiles"],
                    "name_a": m1.get("name", m1["smiles"]),
                    "name_b": m2.get("name", m2["smiles"]),
                    "activity_a": round(m1["activity"], 3),
                    "activity_b": round(m2["activity"], 3),
                    "delta_activity": round(delta, 3),
                    "scaffold": scaffold,
                    "core_mcs": core_smarts,
                    "heavy_atom_diff": atoms2 - atoms1,
                }
                pairs.append(pair_info)

                # Track transformations by scaffold
                key = scaffold
                if key not in transformations:
                    transformations[key] = []
                transformations[key].append({
                    "from": m1["smiles"],
                    "to": m2["smiles"],
                    "delta": delta,
                })

    # Rank pairs by absolute activity change
    pairs.sort(key=lambda x: abs(x["delta_activity"]), reverse=True)

    # Aggregate transformation statistics per scaffold
    scaffold_stats = []
    for scaffold, trans_list in transformations.items():
        deltas = [t["delta"] for t in trans_list]
        scaffold_stats.append({
            "scaffold": scaffold,
            "n_pairs": len(trans_list),
            "mean_delta": round(float(np.mean(deltas)), 3),
            "max_delta": round(float(np.max(deltas)), 3),
            "min_delta": round(float(np.min(deltas)), 3),
            "std_delta": round(float(np.std(deltas)), 3) if len(deltas) > 1 else 0.0,
        })

    # Find top activity-improving transformations
    top_improvements = [p for p in pairs if p["delta_activity"] > 0][:10]
    top_decreases = [p for p in pairs if p["delta_activity"] < 0]
    top_decreases.sort(key=lambda x: x["delta_activity"])
    top_decreases = top_decreases[:5]

    n_scaffolds = len(scaffold_groups)
    using_demo = compounds_csv is None

    summary_lines = [
        f"MMP analysis: {len(parsed)} compounds, {len(pairs)} matched pairs, {n_scaffolds} scaffold(s)",
    ]
    if using_demo:
        summary_lines.append("(Using built-in demo dataset — provide compounds_csv for custom analysis)")
    if top_improvements:
        best = top_improvements[0]
        summary_lines.append(
            f"Best improvement: {best['name_a']} -> {best['name_b']} "
            f"(delta={best['delta_activity']:+.3f})"
        )

    return {
        "summary": "\n".join(summary_lines),
        "n_compounds": len(parsed),
        "n_pairs": len(pairs),
        "n_scaffolds": n_scaffolds,
        "using_demo_data": using_demo,
        "top_improvements": top_improvements,
        "top_decreases": top_decreases,
        "scaffold_stats": scaffold_stats,
        "all_pairs": pairs[:50],  # cap output
    }


@registry.register(
    name="chemistry.scaffold_hop",
    description="Suggest scaffold replacements and bioisosteres for a compound",
    category="chemistry",
    parameters={
        "smiles": "SMILES string for the input compound",
    },
    usage_guide="You want to explore alternative scaffolds for a hit compound — either to improve properties, escape a patent, or find novel chemical matter. Generates bioisosteric replacements for functional groups and suggests scaffold hops based on the Murcko framework.",
)
def scaffold_hop(smiles: str, **kwargs) -> dict:
    """Suggest scaffold replacements and bioisosteric substitutions.

    Extracts the Murcko scaffold, identifies key functional groups, and
    suggests common bioisosteric replacements with rationale.
    """
    smiles = _extract_smiles(smiles)
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        return {"error": "RDKit is required for scaffold hopping. Install with: pip install rdkit", "summary": "RDKit is required for scaffold hopping. Install with: pip install rdkit"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}", "summary": f"Could not parse SMILES: {smiles}"}

    # Extract Murcko scaffold
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smi = Chem.MolToSmiles(scaffold_mol)
        generic_scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
        generic_smi = Chem.MolToSmiles(generic_scaffold)
    except Exception as e:
        scaffold_smi = "N/A"
        generic_smi = "N/A"

    # Identify functional groups via SMARTS matching
    # Each entry: (name, smarts, bioisosteres)
    fg_definitions = [
        ("carboxylic_acid", "[CX3](=O)[OX2H1]", [
            {"replacement": "tetrazole", "smiles_fragment": "c1nnn[nH]1",
             "rationale": "Classic carboxylic acid bioisostere — similar pKa, improved metabolic stability and permeability"},
            {"replacement": "acyl sulfonamide", "smiles_fragment": "C(=O)NS(=O)=O",
             "rationale": "Acidic NH mimics carboxylate — good for oral bioavailability"},
            {"replacement": "hydroxamic acid", "smiles_fragment": "C(=O)NO",
             "rationale": "Maintains H-bond donor/acceptor pattern — also a zinc-binding group"},
        ]),
        ("amide", "[NX3][CX3](=[OX1])[#6]", [
            {"replacement": "sulfonamide", "smiles_fragment": "NS(=O)(=O)",
             "rationale": "Similar geometry and H-bonding — often improved metabolic stability"},
            {"replacement": "urea", "smiles_fragment": "NC(=O)N",
             "rationale": "Additional H-bond donor — can improve target binding"},
            {"replacement": "reversed amide", "smiles_fragment": "C(=O)N (reversed)",
             "rationale": "Switching C(=O)NH to NHC(=O) — changes metabolic soft spot"},
            {"replacement": "1,2,4-oxadiazole", "smiles_fragment": "c1nonc1",
             "rationale": "Planar amide bioisostere — improved metabolic stability"},
        ]),
        ("phenyl", "c1ccccc1", [
            {"replacement": "pyridine", "smiles_fragment": "c1ccncc1",
             "rationale": "Introduces H-bond acceptor — improves solubility and can modulate pKa"},
            {"replacement": "pyrimidine", "smiles_fragment": "c1ncncc1",
             "rationale": "Two nitrogen atoms — further improved solubility vs pyridine"},
            {"replacement": "cyclohexane", "smiles_fragment": "C1CCCCC1",
             "rationale": "sp3-rich replacement — escape flatness, improve Fsp3 and solubility (Lovering)"},
            {"replacement": "thiophene", "smiles_fragment": "c1ccsc1",
             "rationale": "5-membered aromatic — different vector geometry, often similar binding"},
        ]),
        ("ester", "[#6][CX3](=O)[OX2][#6]", [
            {"replacement": "amide", "smiles_fragment": "C(=O)N",
             "rationale": "Much more metabolically stable — standard ester prodrug reversal"},
            {"replacement": "oxadiazole", "smiles_fragment": "c1nonn1",
             "rationale": "Planar ester bioisostere — metabolically stable"},
        ]),
        ("sulfonamide", "[NX3]S(=O)(=O)", [
            {"replacement": "amide", "smiles_fragment": "NC(=O)",
             "rationale": "Simpler, often similar activity — different metabolic profile"},
            {"replacement": "reverse sulfonamide", "smiles_fragment": "S(=O)(=O)N (reversed)",
             "rationale": "Switch N and C sides of sulfonamide"},
        ]),
        ("hydroxyl", "[OX2H]", [
            {"replacement": "fluorine", "smiles_fragment": "F",
             "rationale": "Similar size, H-bond acceptor only — blocks metabolic oxidation site"},
            {"replacement": "amine", "smiles_fragment": "N",
             "rationale": "H-bond donor and acceptor — different pKa profile"},
            {"replacement": "methoxy", "smiles_fragment": "OC",
             "rationale": "Caps the OH — blocks glucuronidation, changes H-bonding"},
        ]),
        ("nitrile", "[CX2]#[NX1]", [
            {"replacement": "isoxazole", "smiles_fragment": "c1ccon1",
             "rationale": "Ring-based CN mimic — similar dipole and H-bond accepting"},
        ]),
        ("fluorine", "[F]", [
            {"replacement": "chlorine", "smiles_fragment": "Cl",
             "rationale": "Larger halogen — increased lipophilicity, different steric profile"},
            {"replacement": "hydrogen", "smiles_fragment": "[H]",
             "rationale": "Remove halogen — simplify molecule, assess fluorine contribution"},
            {"replacement": "trifluoromethyl", "smiles_fragment": "C(F)(F)F",
             "rationale": "Strongly electron-withdrawing — metabolically stable, increases lipophilicity"},
        ]),
    ]

    # Match functional groups
    detected_groups = []
    all_bioisosteres = []

    for fg_name, smarts, bioisosteres in fg_definitions:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            detected_groups.append({
                "group": fg_name,
                "count": len(matches),
                "atom_indices": [list(m) for m in matches],
            })
            for bio in bioisosteres:
                all_bioisosteres.append({
                    "original_group": fg_name,
                    **bio,
                })

    # Scaffold replacement suggestions
    scaffold_replacements = []

    # Detect ring systems in the scaffold
    if scaffold_smi != "N/A":
        scaf_mol = Chem.MolFromSmiles(scaffold_smi)
        if scaf_mol:
            ring_info = scaf_mol.GetRingInfo()
            n_rings = ring_info.NumRings()

            # Common scaffold hops based on ring system
            ring_replacements = {
                "c1ccccc1": [  # benzene
                    ("c1ccncc1", "phenyl -> pyridyl (N-walk around ring)"),
                    ("c1ccoc1", "phenyl -> furanyl (ring contraction)"),
                    ("c1ccsc1", "phenyl -> thiophenyl (5-mem heterocycle)"),
                    ("C1CCCCC1", "phenyl -> cyclohexyl (sp3 escape)"),
                    ("c1cc[nH]c1", "phenyl -> pyrrolyl (electron-rich 5-mem)"),
                ],
                "c1ccncc1": [  # pyridine
                    ("c1ccccc1", "pyridyl -> phenyl (remove N)"),
                    ("c1ncncc1", "pyridyl -> pyrimidyl (add N)"),
                    ("c1ccnnc1", "pyridyl -> pyridazinyl (adjacent N)"),
                ],
                "c1cc[nH]c1": [  # pyrrole
                    ("c1ccoc1", "pyrrolyl -> furanyl"),
                    ("c1ccsc1", "pyrrolyl -> thiophenyl"),
                ],
            }

            for ring_smi, replacements in ring_replacements.items():
                ring_pat = Chem.MolFromSmarts(ring_smi)
                if ring_pat and scaf_mol.HasSubstructMatch(ring_pat):
                    for repl_smi, description in replacements:
                        scaffold_replacements.append({
                            "original_ring": ring_smi,
                            "replacement_ring": repl_smi,
                            "description": description,
                        })

    # Compute properties for context
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)

    property_context = {
        "molecular_weight": round(mw, 1),
        "logp": round(logp, 2),
        "tpsa": round(tpsa, 1),
        "fsp3": round(fsp3, 3),
        "suggestions_for_improvement": [],
    }

    if fsp3 < 0.25:
        property_context["suggestions_for_improvement"].append(
            "Low Fsp3 ({:.2f}) — consider sp3-rich scaffold hops (phenyl->cyclohexyl) to improve solubility".format(fsp3)
        )
    if logp > 4:
        property_context["suggestions_for_improvement"].append(
            "High LogP ({:.1f}) — add heteroatoms or polar groups to improve solubility".format(logp)
        )
    if tpsa < 40:
        property_context["suggestions_for_improvement"].append(
            "Low TPSA ({:.0f}) — may have poor solubility; consider adding H-bond acceptors".format(tpsa)
        )

    # Summary
    summary_lines = [
        f"Scaffold analysis for {smiles}",
        f"Murcko scaffold: {scaffold_smi}",
        f"Generic framework: {generic_smi}",
        f"Detected functional groups: {', '.join(g['group'] for g in detected_groups) if detected_groups else 'none identified'}",
        f"Bioisostere suggestions: {len(all_bioisosteres)}",
        f"Scaffold hop options: {len(scaffold_replacements)}",
    ]

    return {
        "summary": "\n".join(summary_lines),
        "input_smiles": smiles,
        "murcko_scaffold": scaffold_smi,
        "generic_framework": generic_smi,
        "detected_functional_groups": detected_groups,
        "bioisostere_suggestions": all_bioisosteres,
        "scaffold_replacements": scaffold_replacements,
        "property_context": property_context,
    }


@registry.register(
    name="chemistry.pubchem_lookup",
    description="Look up compound data from PubChem by name or SMILES",
    category="chemistry",
    parameters={
        "query": "Compound name or SMILES string",
        "query_type": "Type of query: 'name' or 'smiles' (default 'name')",
    },
    usage_guide="You need compound information (structure, properties, synonyms, CID) from PubChem. Use when identifying a compound by name or validating a SMILES string. Returns canonical SMILES, physicochemical properties, and identifiers.",
)
def pubchem_lookup(query: str, query_type: str = "name", **kwargs) -> dict:
    """Look up compound data from PubChem PUG REST API.

    Supports lookup by compound name or SMILES string. Returns CID, canonical
    SMILES, molecular properties, and synonyms.
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    # Step 1: Resolve query to CID
    if query_type == "smiles":
        lookup_url = f"{base_url}/compound/smiles/JSON"
        # Use POST for SMILES to handle special characters
        resp, error = request(
            "POST",
            lookup_url,
            data={"smiles": query},
            timeout=10,
            retries=2,
            raise_for_status=False,
        )
        if error:
            return {"error": f"HTTP error: {error}", "summary": f"PubChem lookup failed: {error}"}
    else:
        # URL-encode the compound name
        import urllib.parse
        encoded_query = urllib.parse.quote(query, safe="")
        lookup_url = f"{base_url}/compound/name/{encoded_query}/JSON"
        resp, error = request(
            "GET",
            lookup_url,
            timeout=10,
            retries=2,
            raise_for_status=False,
        )
        if error:
            return {"error": f"HTTP error: {error}", "summary": f"PubChem lookup failed: {error}"}

    if resp.status_code == 404:
        return {
            "error": f"Compound not found: {query}",
            "summary": f"PubChem: no compound found for '{query}' (query_type={query_type})",
        }
    if resp.status_code != 200:
        return {
            "error": f"PubChem API error (HTTP {resp.status_code})",
            "summary": f"PubChem lookup failed with status {resp.status_code}",
        }

    try:
        data = resp.json()
    except Exception:
        return {"error": "Failed to parse PubChem response", "summary": "PubChem returned invalid JSON"}

    # Extract CID
    compounds = data.get("PC_Compounds", [])
    if not compounds:
        return {"error": "No compound data in response", "summary": f"PubChem returned empty result for '{query}'"}

    compound = compounds[0]
    cid = compound.get("id", {}).get("id", {}).get("cid")

    if not cid:
        return {"error": "Could not extract CID", "summary": "PubChem response missing CID"}

    # Step 2: Get properties
    props_url = (
        f"{base_url}/compound/cid/{cid}/property/"
        "MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,"
        "XLogP,ExactMass,TPSA,HBondDonorCount,HBondAcceptorCount,"
        "RotatableBondCount,HeavyAtomCount,Complexity/JSON"
    )
    props_resp, props_error = request(
        "GET",
        props_url,
        timeout=10,
        retries=2,
        raise_for_status=False,
    )
    if props_error:
        props_data = {}
    else:
        try:
            props_data = props_resp.json() if props_resp.status_code == 200 else {}
        except Exception:
            props_data = {}

    properties = {}
    prop_table = props_data.get("PropertyTable", {}).get("Properties", [])
    if prop_table:
        p = prop_table[0]
        # PubChem may return SMILES as "CanonicalSMILES", "SMILES", or "ConnectivitySMILES"
        canonical = p.get("CanonicalSMILES") or p.get("SMILES") or p.get("ConnectivitySMILES")
        isomeric = p.get("IsomericSMILES") or canonical
        properties = {
            "cid": p.get("CID"),
            "molecular_formula": p.get("MolecularFormula"),
            "molecular_weight": p.get("MolecularWeight"),
            "canonical_smiles": canonical,
            "isomeric_smiles": isomeric,
            "xlogp": p.get("XLogP"),
            "exact_mass": p.get("ExactMass"),
            "tpsa": p.get("TPSA"),
            "hbd": p.get("HBondDonorCount"),
            "hba": p.get("HBondAcceptorCount"),
            "rotatable_bonds": p.get("RotatableBondCount"),
            "heavy_atoms": p.get("HeavyAtomCount"),
            "complexity": p.get("Complexity"),
        }

    # Step 3: Get synonyms
    synonyms_url = f"{base_url}/compound/cid/{cid}/synonyms/JSON"
    synonyms = []
    syn_resp, syn_error = request(
        "GET",
        synonyms_url,
        timeout=10,
        retries=2,
        raise_for_status=False,
    )
    if not syn_error and syn_resp.status_code == 200:
        try:
            syn_data = syn_resp.json()
            syn_list = syn_data.get("InformationList", {}).get("Information", [])
            if syn_list:
                synonyms = syn_list[0].get("Synonym", [])[:20]  # cap at 20
        except Exception:
            pass

    # Build summary
    canonical = properties.get("canonical_smiles", "N/A")
    mw = properties.get("molecular_weight", "N/A")
    formula = properties.get("molecular_formula", "N/A")
    xlogp = properties.get("xlogp", "N/A")

    summary_lines = [
        f"PubChem: {query} (CID {cid})",
        f"Formula: {formula}, MW: {mw}, XLogP: {xlogp}",
        f"SMILES: {canonical}",
    ]
    if synonyms:
        summary_lines.append(f"Also known as: {', '.join(synonyms[:5])}")

    return {
        "summary": "\n".join(summary_lines),
        "cid": cid,
        "canonical_smiles": properties.get("canonical_smiles"),
        "properties": properties,
        "synonyms": synonyms,
        "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
    }


# ─── Retrosynthetic transforms (SMARTS) ─────────────────────────
# Each transform: (name, product_smarts, reactant_smarts_list, reagents, conditions)
_RETRO_TRANSFORMS = [
    {
        "name": "Amide bond disconnection",
        "description": "Disconnect C(=O)-N amide bond → carboxylic acid + amine",
        "product_smarts": "[C:1](=[O:2])-[N:3]",
        "reagents": ["HATU", "DIPEA"],
        "conditions": "Amide coupling, DMF, RT, 12h",
        "reaction_class": "amide_coupling",
    },
    {
        "name": "Suzuki coupling",
        "description": "Disconnect Ar-Ar biaryl bond → aryl boronic acid + aryl halide",
        "product_smarts": "[c:1]-[c:2]",
        "reagents": ["Pd(PPh3)4", "K2CO3"],
        "conditions": "Suzuki coupling, dioxane/H2O, 80°C, 16h",
        "reaction_class": "cross_coupling",
    },
    {
        "name": "Ester hydrolysis",
        "description": "Disconnect C(=O)-O ester bond → carboxylic acid + alcohol",
        "product_smarts": "[C:1](=[O:2])-[O:3][C:4]",
        "reagents": ["DCC", "DMAP"],
        "conditions": "Esterification, DCM, RT, 4h",
        "reaction_class": "esterification",
    },
    {
        "name": "Reductive amination",
        "description": "Disconnect C-N bond adjacent to C-H → aldehyde/ketone + amine",
        "product_smarts": "[C:1]-[NH:2]",
        "reagents": ["NaBH3CN", "AcOH"],
        "conditions": "Reductive amination, MeOH, RT, 16h",
        "reaction_class": "reductive_amination",
    },
    {
        "name": "N-alkylation",
        "description": "Disconnect N-C(sp3) bond → amine + alkyl halide",
        "product_smarts": "[N:1]-[CH2:2]",
        "reagents": ["K2CO3"],
        "conditions": "N-alkylation, DMF, 60°C, 12h",
        "reaction_class": "alkylation",
    },
    {
        "name": "Ether formation (Williamson)",
        "description": "Disconnect C-O-C ether bond → alcohol + alkyl halide",
        "product_smarts": "[C:1]-[O:2]-[C:3]",
        "reagents": ["NaH"],
        "conditions": "Williamson ether synthesis, THF, 0°C→RT, 6h",
        "reaction_class": "etherification",
    },
    {
        "name": "Sulfonamide formation",
        "description": "Disconnect S(=O)(=O)-N bond → sulfonyl chloride + amine",
        "product_smarts": "[S:1](=[O:2])(=[O:3])-[N:4]",
        "reagents": ["Et3N"],
        "conditions": "Sulfonamide coupling, DCM, 0°C→RT, 4h",
        "reaction_class": "sulfonamide_formation",
    },
    {
        "name": "Urea formation",
        "description": "Disconnect N-C(=O)-N urea → isocyanate + amine",
        "product_smarts": "[N:1]-[C:2](=[O:3])-[N:4]",
        "reagents": ["CDI or triphosgene"],
        "conditions": "Urea formation, DCM, RT, 12h",
        "reaction_class": "urea_formation",
    },
]


@registry.register(
    name="chemistry.retrosynthesis",
    description="Plan retrosynthetic routes for a target molecule — uses IBM RXN API if configured, otherwise heuristic SMARTS-based disconnections",
    category="chemistry",
    parameters={
        "smiles": "SMILES string of the target molecule",
        "max_steps": "Maximum retrosynthetic steps (default 3)",
    },
    usage_guide="You want to plan a synthetic route to make a target compound. Use for synthesis feasibility assessment, identifying key disconnections, and suggesting reagents/conditions. Provides heuristic retrosynthetic analysis using common transforms; optionally uses IBM RXN API if an API key is configured.",
)
def retrosynthesis(smiles: str, max_steps: int = 3, **kwargs) -> dict:
    """Plan retrosynthetic routes for a target molecule.

    Attempts the IBM RXN API first (if api.ibm_rxn_key is configured),
    then falls back to a heuristic RDKit-based retrosynthesis using
    common disconnection transforms.
    """
    smiles = _extract_smiles(smiles)

    # Try IBM RXN API first
    session = kwargs.get("_session", None)
    api_key = None
    if session and hasattr(session, "config"):
        api_key = session.config.get("api.ibm_rxn_key", None)

    if api_key:
        result = _retrosynthesis_ibm_rxn(smiles, max_steps, api_key)
        if result and "error" not in result:
            return result

    # Fall back to heuristic RDKit retrosynthesis
    return _retrosynthesis_heuristic(smiles, max_steps)


def _retrosynthesis_ibm_rxn(smiles: str, max_steps: int, api_key: str) -> dict:
    """Call IBM RXN API for retrosynthesis prediction."""
    import time

    base_url = "https://rxn.res.ibm.com/rxn/api/api/v1"
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }

    # Submit retrosynthesis prediction
    resp, error = request(
        "POST",
        f"{base_url}/retrosynthesis/predict",
        json={"content": smiles, "maxSteps": max_steps},
        headers=headers,
        timeout=30,
        retries=2,
        raise_for_status=False,
    )
    if error:
        return {"error": f"IBM RXN API request failed: {error}", "summary": f"IBM RXN API request failed: {error}"}
    if resp.status_code != 200:
        return {"error": f"IBM RXN API returned status {resp.status_code}", "summary": f"IBM RXN API returned status {resp.status_code}"}
    try:
        prediction_id = resp.json().get("prediction_id")
    except Exception:
        return {"error": "IBM RXN API returned invalid JSON", "summary": "IBM RXN API returned invalid JSON"}
    if not prediction_id:
        return {"error": "IBM RXN API did not return a prediction ID", "summary": "IBM RXN API did not return a prediction ID"}
    # Poll for results (up to 60 seconds)
    for _ in range(12):
        time.sleep(5)
        poll_resp, poll_error = request(
            "GET",
            f"{base_url}/retrosynthesis/results/{prediction_id}",
            headers=headers,
            timeout=15,
            retries=1,
            raise_for_status=False,
        )
        if poll_error:
            import logging
            logging.getLogger("ct.tools.chemistry").debug(
                "IBM RXN poll attempt failed: %s", poll_error,
            )
            continue
        if poll_resp.status_code == 200:
            try:
                data = poll_resp.json()
            except Exception:
                continue
            status = data.get("status", "")
            if status == "SUCCESS":
                return _parse_ibm_rxn_results(smiles, data)
            if status == "FAILED":
                return {"error": "IBM RXN retrosynthesis failed", "summary": "IBM RXN retrosynthesis failed"}
    return {"error": "IBM RXN API timed out waiting for results", "summary": "IBM RXN API timed out waiting for results"}
def _parse_ibm_rxn_results(smiles: str, data: dict) -> dict:
    """Parse IBM RXN API retrosynthesis results into standard format."""
    routes = []
    retro_routes = data.get("retrosynthetic_paths", [])

    for i, route in enumerate(retro_routes):
        steps = []
        for step in route.get("steps", []):
            steps.append({
                "reaction_smiles": step.get("reaction", ""),
                "reactants": step.get("reactants", []),
                "confidence": step.get("confidence", 0.0),
            })
        routes.append({
            "route_id": i + 1,
            "n_steps": len(steps),
            "steps": steps,
            "confidence": route.get("confidence", 0.0),
        })

    routes.sort(key=lambda r: r["n_steps"])
    shortest = routes[0]["n_steps"] if routes else 0

    return {
        "summary": f"Retrosynthesis for {smiles}: {len(routes)} routes found via IBM RXN, "
                   f"shortest is {shortest} steps",
        "target": smiles,
        "source": "ibm_rxn",
        "n_routes": len(routes),
        "routes": routes,
    }


def _retrosynthesis_heuristic(smiles: str, max_steps: int) -> dict:
    """Heuristic retrosynthesis using RDKit SMARTS-based disconnections."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}", "summary": f"Could not parse SMILES: {smiles}"}

    # Find applicable disconnections
    disconnections = []
    for transform in _RETRO_TRANSFORMS:
        pattern = Chem.MolFromSmarts(transform["product_smarts"])
        if pattern is None:
            continue
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            disconnections.append({
                "transform_name": transform["name"],
                "description": transform["description"],
                "n_sites": len(matches),
                "atom_indices": [list(m) for m in matches[:3]],  # cap at 3
                "reagents": transform["reagents"],
                "conditions": transform["conditions"],
                "reaction_class": transform["reaction_class"],
            })

    if not disconnections:
        return {
            "summary": f"[HEURISTIC FALLBACK] Retrosynthesis for {smiles}: no heuristic disconnections found — "
                       "molecule may require specialized chemistry. Configure api.ibm_rxn_key for AI-powered retrosynthesis.",
            "target": smiles,
            "source": "heuristic",
            "n_routes": 0,
            "routes": [],
            "disconnections": [],
        }

    # Build routes: each disconnection is a potential first step
    # For multi-step, try to fragment further (simplified: just report single-step disconnections
    # but note that BRICS can provide deeper fragmentation)
    routes = []
    for i, disc in enumerate(disconnections):
        route_steps = [{
            "step": 1,
            "transform": disc["transform_name"],
            "description": disc["description"],
            "reagents": disc["reagents"],
            "conditions": disc["conditions"],
            "n_disconnection_sites": disc["n_sites"],
        }]
        routes.append({
            "route_id": i + 1,
            "strategy": disc["transform_name"],
            "n_steps": 1,
            "steps": route_steps,
            "reaction_class": disc["reaction_class"],
        })

    # BRICS decomposition for deeper analysis
    brics_fragments = []
    try:
        from rdkit.Chem import BRICS
        frags = BRICS.BRICSDecompose(mol, returnMols=False)
        brics_fragments = list(frags)[:10]  # cap output
    except Exception:
        pass

    # Add a BRICS-based route if fragments found
    if brics_fragments and len(brics_fragments) > 1:
        brics_steps = []
        for j, frag in enumerate(brics_fragments):
            brics_steps.append({
                "step": j + 1,
                "fragment": frag,
                "description": f"BRICS fragment {j + 1}",
            })
        routes.append({
            "route_id": len(routes) + 1,
            "strategy": "BRICS full decomposition",
            "n_steps": len(brics_fragments),
            "steps": brics_steps,
            "reaction_class": "brics",
        })

    # Molecular properties for context
    mw = Descriptors.MolWt(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)

    # Sort routes by step count
    routes.sort(key=lambda r: r["n_steps"])
    shortest = routes[0]["n_steps"] if routes else 0

    return {
        "summary": f"[HEURISTIC FALLBACK] Retrosynthesis for {formula} ({smiles}): {len(routes)} routes found "
                   f"via SMARTS-based disconnection (not AI-predicted). Configure api.ibm_rxn_key for more accurate routes.",
        "target": smiles,
        "formula": formula,
        "molecular_weight": round(mw, 1),
        "source": "heuristic",
        "n_routes": len(routes),
        "routes": routes,
        "disconnections": disconnections,
        "brics_fragments": brics_fragments,
    }


# ─── Pharmacophore feature SMARTS definitions ──────────────────
_PHARMACOPHORE_FEATURES = {
    "HBD": {
        "name": "Hydrogen Bond Donor",
        "smarts": ["[#7!H0&!$(N-[SX4](=O)(=O)[CX4](F)(F)F)]", "[#8!H0&!$([OH][C,S,P]=O)]", "[#16!H0]"],
    },
    "HBA": {
        "name": "Hydrogen Bond Acceptor",
        "smarts": ["[#7&!$([nH])&!$(N-N=O)]", "[$([O])&!$([OX2](C)C=O)]", "[#16&X2]"],
    },
    "Aromatic": {
        "name": "Aromatic Ring",
        "smarts": ["a1aaaaa1", "a1aaaa1"],
    },
    "Hydrophobic": {
        "name": "Hydrophobic",
        "smarts": ["[CH2X4,CH1X4,CH0X4]", "[$([cX3](:*):*)&!$([cX3](-[OH])-[OH])]"],
    },
    "PosIonizable": {
        "name": "Positive Ionizable",
        "smarts": ["[+,+2,+3,+4]", "[$([NX3&!$([NX3]-O)](-C)(-C)-C)]", "[$(n1cc[nH]c1)]"],
    },
    "NegIonizable": {
        "name": "Negative Ionizable",
        "smarts": ["[-,-2,-3,-4]", "[$([OH]-[CX3]=[OX1])]", "[$([OH]-[SX4](=[OX1])(=[OX1]))]"],
    },
}


@registry.register(
    name="chemistry.pharmacophore",
    description="Generate a pharmacophore model from a set of active compounds identifying common molecular features",
    category="chemistry",
    parameters={
        "smiles_list": "List of SMILES strings for active compounds",
        "method": "Analysis method: 'common_features' (default) or 'fingerprints'",
    },
    usage_guide="You have a set of active compounds and want to identify the common pharmacophoric features that drive activity. Use for understanding SAR, virtual screening, and lead optimization. Identifies shared HBD, HBA, aromatic, hydrophobic, and ionizable features across the compound set.",
)
def pharmacophore(smiles_list: list = None, method: str = "common_features", **kwargs) -> dict:
    """Generate a pharmacophore model from a set of active compounds.

    Identifies common pharmacophore features (HBD, HBA, Aromatic, Hydrophobic,
    PosIonizable, NegIonizable) across the compound set and optionally generates
    2D pharmacophore fingerprints for consensus scoring.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors

    if not smiles_list or len(smiles_list) < 2:
        return {
            "error": "Need at least 2 SMILES strings",
            "summary": "Pharmacophore analysis requires at least 2 compounds",
        }

    # Resolve any drug names to SMILES
    resolved_list = []
    for smi in smiles_list:
        resolved_list.append(_extract_smiles(smi))
    smiles_list = resolved_list

    # Parse molecules
    mols = []
    valid_smiles = []
    invalid = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.append(smi)
        else:
            mols.append(mol)
            valid_smiles.append(smi)

    if len(mols) < 2:
        return {
            "error": f"Only {len(mols)} valid molecule(s) — need at least 2",
            "summary": "Insufficient valid molecules for pharmacophore analysis",
            "invalid_smiles": invalid,
        }

    n_compounds = len(mols)

    # Step 1: Detect pharmacophore features per molecule
    per_molecule_features = []  # list of dicts: smiles -> {feature_type: count}

    for i, mol in enumerate(mols):
        mol_features = {}
        for feat_type, feat_def in _PHARMACOPHORE_FEATURES.items():
            count = 0
            for smarts_str in feat_def["smarts"]:
                pattern = Chem.MolFromSmarts(smarts_str)
                if pattern is not None:
                    matches = mol.GetSubstructMatches(pattern)
                    count += len(matches)
            mol_features[feat_type] = count
        per_molecule_features.append({
            "smiles": valid_smiles[i],
            "features": mol_features,
        })

    # Step 2: Identify common features (present in all molecules)
    feature_types = list(_PHARMACOPHORE_FEATURES.keys())
    common_features = []
    feature_distribution = {}

    for feat_type in feature_types:
        counts = [mf["features"][feat_type] for mf in per_molecule_features]
        min_count = min(counts)
        max_count = max(counts)
        mean_count = sum(counts) / len(counts)
        # Feature is "common" if present in all molecules
        present_in = sum(1 for c in counts if c > 0)
        frequency = present_in / n_compounds

        feature_distribution[feat_type] = {
            "name": _PHARMACOPHORE_FEATURES[feat_type]["name"],
            "min_count": min_count,
            "max_count": max_count,
            "mean_count": round(mean_count, 1),
            "present_in_n": present_in,
            "frequency": round(frequency, 3),
        }

        if min_count > 0:
            common_features.append({
                "type": feat_type,
                "name": _PHARMACOPHORE_FEATURES[feat_type]["name"],
                "min_count": min_count,
                "conserved": min_count == max_count,
                "frequency": 1.0,
            })

    # Step 3: 2D pharmacophore fingerprints (if method includes fingerprints)
    pharm_fp_similarity = None
    if method in ("fingerprints", "both"):
        try:
            from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate

            factory = Gobbi_Pharm2D.factory
            fps = []
            for mol in mols:
                fp = Generate.Gen2DFingerprint(mol, factory)
                fps.append(fp)

            # Pairwise Tanimoto similarity
            sim_sum = 0.0
            sim_count = 0
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    sim_sum += sim
                    sim_count += 1

            pharm_fp_similarity = round(sim_sum / sim_count, 3) if sim_count > 0 else 0.0
        except Exception:
            pharm_fp_similarity = None  # Gobbi_Pharm2D not available

    # Step 4: Consensus score
    # Based on: fraction of features that are common + consistency of counts
    if feature_types:
        common_frac = len(common_features) / len(feature_types)
    else:
        common_frac = 0.0

    # Weight by how conserved the counts are (lower variance = higher consensus)
    variance_scores = []
    for feat_type in feature_types:
        counts = [mf["features"][feat_type] for mf in per_molecule_features]
        if max(counts) > 0:
            normalized_range = (max(counts) - min(counts)) / max(counts)
            variance_scores.append(1.0 - normalized_range)

    consistency = sum(variance_scores) / len(variance_scores) if variance_scores else 0.0
    consensus_score = round((common_frac * 0.6 + consistency * 0.4), 3)

    # Build summary
    common_desc = []
    for cf in common_features:
        common_desc.append(f"{cf['min_count']} {cf['name']}")

    summary_parts = [
        f"Pharmacophore from {n_compounds} compounds: "
        f"{len(common_features)} common features",
    ]
    if common_desc:
        summary_parts[0] += f" ({', '.join(common_desc)})"
    summary_parts.append(f"Consensus score: {consensus_score}")
    if pharm_fp_similarity is not None:
        summary_parts.append(f"Mean pharmacophore fingerprint similarity: {pharm_fp_similarity}")

    result = {
        "summary": "\n".join(summary_parts),
        "n_compounds": n_compounds,
        "n_valid": len(mols),
        "common_features": common_features,
        "feature_distribution": feature_distribution,
        "per_molecule_features": per_molecule_features,
        "consensus_score": consensus_score,
        "method": method,
    }

    if invalid:
        result["invalid_smiles"] = invalid
    if pharm_fp_similarity is not None:
        result["pharmacophore_fp_similarity"] = pharm_fp_similarity

    return result
