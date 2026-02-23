"""
Design tools: compound modification suggestions, med-chem optimization.

Provides AI-guided medicinal chemistry recommendations using RDKit-based
property calculations, bioisosteric replacement rules, and Lipinski/Veber
scoring.
"""

from ct.tools import registry


# Common medicinal chemistry transformations: (SMARTS_pattern, SMARTS_replacement, description)
# Each entry: (name, smarts_from, list_of_replacements)
_MEDCHEM_TRANSFORMS = [
    # Halogen walks
    ("F_to_Cl", "[cH0:1][F:2]", "[cH0:1][Cl]", "F->Cl: increase lipophilicity and steric bulk"),
    ("F_to_H", "[cH0:1][F:2]", "[cH:1]", "F->H: remove halogen, reduce MW"),
    ("Cl_to_F", "[cH0:1][Cl:2]", "[cH0:1][F]", "Cl->F: reduce lipophilicity, metabolic block"),
    # Alkyl modifications
    ("Me_to_Et", "[CH3:1]([#6:2])", "[CH2:1]([#6:2])C", "Me->Et: increase steric bulk, explore SAR"),
    ("OH_to_OMe", "[c:1][OH:2]", "[c:1]OC", "OH->OMe: cap phenol, block glucuronidation"),
    ("OMe_to_OH", "[c:1][O:2][CH3]", "[c:1][OH]", "OMe->OH: add H-bond donor, improve solubility"),
    # N-modifications
    ("NH_to_NMe", "[NH2:1]", "[NH:1]C", "NH2->NHMe: reduce basicity, improve metabolic stability"),
    ("NMe_to_NH", "[NH:1]([CH3])", "[NH2:1]", "NHMe->NH2: simplify, add H-bond donor"),
    # Ring modifications
    ("phenyl_to_pyridine", "[c:1]1[c:2][c:3][c:4][c:5][cH:6]1", "[c:1]1[c:2][c:3][c:4][c:5][n:6]1",
     "phenyl->pyridyl: improve solubility, add H-bond acceptor"),
]


def _compute_properties(mol) -> dict:
    """Compute drug-relevant molecular properties."""
    from rdkit.Chem import Descriptors, rdMolDescriptors

    return {
        "mw": round(Descriptors.MolWt(mol), 1),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "tpsa": round(Descriptors.TPSA(mol), 1),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "rings": Descriptors.RingCount(mol),
        "aromatic_rings": Descriptors.NumAromaticRings(mol),
        "fsp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
    }


def _lipinski_violations(props: dict) -> int:
    """Count Lipinski Rule-of-5 violations."""
    return sum([
        props["mw"] > 500,
        props["logp"] > 5,
        props["hbd"] > 5,
        props["hba"] > 10,
    ])


def _veber_violations(props: dict) -> int:
    """Count Veber oral bioavailability rule violations."""
    return sum([
        props["tpsa"] > 140,
        props["rotatable_bonds"] > 10,
    ])


def _score_for_objective(parent_props: dict, child_props: dict,
                         objective: str) -> float:
    """Score a modification relative to the parent, for a given objective.

    Returns a float where higher is better (range roughly -1 to +1).
    """
    if objective == "potency":
        # Favour: lower MW, moderate LogP (2-4), more H-bond interactions
        logp_score = 1.0 - abs(child_props["logp"] - 3.0) / 5.0
        mw_score = (parent_props["mw"] - child_props["mw"]) / 100.0
        return round(logp_score * 0.5 + mw_score * 0.3 + 0.2, 3)

    elif objective == "selectivity":
        # Favour: more specific interactions (more HBD/HBA), higher TPSA
        hb_delta = (child_props["hbd"] + child_props["hba"]) - (parent_props["hbd"] + parent_props["hba"])
        tpsa_delta = (child_props["tpsa"] - parent_props["tpsa"]) / 50.0
        return round(hb_delta * 0.3 + tpsa_delta * 0.4 + 0.3, 3)

    elif objective == "admet":
        # Favour: fewer Lipinski/Veber violations, moderate LogP
        lip = _lipinski_violations(child_props)
        veb = _veber_violations(child_props)
        penalty = lip * 0.25 + veb * 0.25
        logp_bonus = 1.0 - abs(child_props["logp"] - 2.5) / 5.0
        return round(max(0, 1.0 - penalty) * 0.6 + logp_bonus * 0.4, 3)

    elif objective == "solubility":
        # Favour: lower LogP, higher TPSA, higher Fsp3
        logp_score = max(0, (parent_props["logp"] - child_props["logp"]) / 3.0)
        tpsa_score = max(0, (child_props["tpsa"] - parent_props["tpsa"]) / 40.0)
        fsp3_score = max(0, (child_props["fsp3"] - parent_props["fsp3"]))
        return round(logp_score * 0.4 + tpsa_score * 0.3 + fsp3_score * 0.3, 3)

    elif objective == "metabolic_stability":
        # Favour: fewer metabolic soft spots (reduce ArOH, add F-blocks)
        logp_score = 1.0 - abs(child_props["logp"] - 2.0) / 5.0
        mw_penalty = max(0, (child_props["mw"] - 500) / 200.0)
        rotbond_penalty = max(0, (child_props["rotatable_bonds"] - 7) / 5.0)
        return round(logp_score * 0.4 - mw_penalty * 0.3 - rotbond_penalty * 0.3, 3)

    # Default: balanced score
    return 0.5


@registry.register(
    name="design.suggest_modifications",
    description="Suggest medicinal chemistry modifications to improve a compound's properties",
    category="design",
    parameters={
        "smiles": "Input compound SMILES string",
        "objective": "Optimization goal: potency, selectivity, admet, solubility, metabolic_stability (default: potency)",
        "n_suggestions": "Number of suggestions to return (default 5)",
    },
    usage_guide=(
        "You have a hit or lead compound and want to generate ideas for "
        "structural modifications to improve potency, selectivity, ADMET, "
        "solubility, or metabolic stability. Returns modified SMILES with "
        "property comparisons and medicinal chemistry rationale."
    ),
)
def suggest_modifications(smiles: str, objective: str = "potency",
                          n_suggestions: int = 5, **kwargs) -> dict:
    """Suggest medicinal chemistry modifications for a compound.

    Applies bioisosteric replacements and common med-chem transforms using
    RDKit reaction SMARTS, then scores each modification against the
    specified objective.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors
    except ImportError:
        return {"error": "RDKit is required for compound modification suggestions (pip install rdkit)", "summary": "RDKit is required for compound modification suggestions (pip install rdkit)"}
    valid_objectives = ("potency", "selectivity", "admet", "solubility", "metabolic_stability")
    if objective not in valid_objectives:
        return {"error": f"Unknown objective '{objective}'. Choose from: {', '.join(valid_objectives)}", "summary": f"Unknown objective '{objective}'. Choose from: {', '.join(valid_objectives)}"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}", "summary": f"Could not parse SMILES: {smiles}"}

    parent_props = _compute_properties(mol)
    parent_violations = _lipinski_violations(parent_props)
    canonical = Chem.MolToSmiles(mol)

    suggestions = []
    seen_smiles = {canonical}

    # Apply each transformation
    for name, smarts_from, smarts_to, description in _MEDCHEM_TRANSFORMS:
        try:
            rxn_smarts = f"[{smarts_from}]>>[{smarts_to}]"
            # Use ReplaceSubstructs instead of reaction for reliability
            pattern = Chem.MolFromSmarts(smarts_from)
            if pattern is None:
                continue
            if not mol.HasSubstructMatch(pattern):
                continue

            # Attempt reaction-based transform
            rxn = AllChem.ReactionFromSmarts(f"{smarts_from}>>{smarts_to}")
            if rxn is None:
                continue

            products = rxn.RunReactants((mol,))
            for product_set in products:
                for product in product_set:
                    try:
                        Chem.SanitizeMol(product)
                        product_smi = Chem.MolToSmiles(product)
                        if product_smi in seen_smiles:
                            continue
                        seen_smiles.add(product_smi)

                        child_props = _compute_properties(product)
                        child_violations = _lipinski_violations(child_props)

                        score = _score_for_objective(parent_props, child_props, objective)

                        # Property deltas
                        deltas = {
                            "mw": round(child_props["mw"] - parent_props["mw"], 1),
                            "logp": round(child_props["logp"] - parent_props["logp"], 2),
                            "hbd": child_props["hbd"] - parent_props["hbd"],
                            "hba": child_props["hba"] - parent_props["hba"],
                            "tpsa": round(child_props["tpsa"] - parent_props["tpsa"], 1),
                        }

                        suggestions.append({
                            "smiles": product_smi,
                            "transform": name,
                            "rationale": description,
                            "score": score,
                            "properties": child_props,
                            "property_deltas": deltas,
                            "lipinski_violations": child_violations,
                        })
                    except Exception:
                        continue
        except Exception:
            continue

    # Also add simple functional group additions if we have few suggestions
    _simple_additions = [
        ("add_F", "F", "Add fluorine — metabolic blocker, minimal size increase"),
        ("add_OH", "O", "Add hydroxyl — improve solubility, add H-bond donor"),
        ("add_NH2", "N", "Add amine — add H-bond donor, potential salt formation"),
        ("add_Me", "C", "Add methyl — explore steric effects, fill hydrophobic pocket"),
    ]

    for add_name, add_atom, add_desc in _simple_additions:
        if len(suggestions) >= n_suggestions * 3:
            break
        try:
            # Find an aromatic carbon that could accept a substituent
            pattern = Chem.MolFromSmarts("[cH]")
            if pattern and mol.HasSubstructMatch(pattern):
                matches = mol.GetSubstructMatches(pattern)
                if matches:
                    from rdkit.Chem import RWMol
                    for match in matches[:1]:  # just first match
                        rw = RWMol(mol)
                        new_idx = rw.AddAtom(Chem.Atom(add_atom))
                        rw.AddBond(match[0], new_idx, Chem.BondType.SINGLE)
                        try:
                            Chem.SanitizeMol(rw)
                            new_smi = Chem.MolToSmiles(rw)
                            if new_smi in seen_smiles:
                                continue
                            seen_smiles.add(new_smi)

                            new_mol = Chem.MolFromSmiles(new_smi)
                            if new_mol is None:
                                continue

                            child_props = _compute_properties(new_mol)
                            child_violations = _lipinski_violations(child_props)
                            score = _score_for_objective(parent_props, child_props, objective)

                            deltas = {
                                "mw": round(child_props["mw"] - parent_props["mw"], 1),
                                "logp": round(child_props["logp"] - parent_props["logp"], 2),
                                "hbd": child_props["hbd"] - parent_props["hbd"],
                                "hba": child_props["hba"] - parent_props["hba"],
                                "tpsa": round(child_props["tpsa"] - parent_props["tpsa"], 1),
                            }

                            suggestions.append({
                                "smiles": new_smi,
                                "transform": add_name,
                                "rationale": add_desc,
                                "score": score,
                                "properties": child_props,
                                "property_deltas": deltas,
                                "lipinski_violations": child_violations,
                            })
                        except Exception:
                            continue
        except Exception:
            continue

    # Sort by score descending, take top n
    suggestions.sort(key=lambda s: s["score"], reverse=True)
    top = suggestions[:n_suggestions]

    if not top:
        return {
            "summary": f"No modifications found for {smiles} (no applicable transforms matched)",
            "parent_smiles": canonical,
            "parent_properties": parent_props,
            "suggestions": [],
        }

    # Build summary
    best = top[0]
    logp_delta = best["property_deltas"]["logp"]
    delta_str = f"LogP {logp_delta:+.2f}" if logp_delta != 0 else "similar LogP"

    return {
        "summary": (
            f"{len(top)} modification(s) suggested for {canonical[:40]} "
            f"(objective={objective}): top suggestion {best['transform']} ({delta_str})"
        ),
        "parent_smiles": canonical,
        "parent_properties": parent_props,
        "parent_lipinski_violations": parent_violations,
        "objective": objective,
        "suggestions": top,
    }
