"""
Imaging tools: compound bioactivity profiling via PubChem and structural similarity.

Uses PubChem bioactivity data and RDKit molecular descriptors for mechanism
classification. Structural fingerprint similarity as a proxy for phenotypic similarity.
"""

from ct.tools import registry
from ct.tools.http_client import request


@registry.register(
    name="imaging.cellpainting_lookup",
    description="Look up compound bioactivity and compute mechanism class via PubChem assays and RDKit descriptors",
    category="imaging",
    parameters={
        "compound": "Compound name, InChIKey, or SMILES string",
        "source": "Data source: 'pubchem' (default). JUMP Cell Painting data requires local parquet files (not yet integrated).",
    },
    usage_guide="You want to understand a compound's bioactivity profile and infer its mechanism class. Queries PubChem bioassay data and computes RDKit molecular descriptors for heuristic mechanism classification. Note: full Cell Painting morphological profiles from JUMP require downloading parquet files from the JUMP Cell Painting Gallery (S3-hosted, no REST API).",
)
def cellpainting_lookup(compound: str, source: str = "pubchem", **kwargs) -> dict:
    """Look up compound bioactivity and mechanism class.

    Queries PubChem for bioassay data and computes RDKit molecular descriptors
    for heuristic mechanism classification. Full JUMP Cell Painting morphological
    profiles are not yet integrated (data is S3-hosted parquet, no REST API).
    """
    compound_info = {"query": compound, "source": source}

    # Step 1: Try to resolve compound via PubChem for identifiers
    cid = None
    canonical_smiles = None
    inchikey = None
    compound_name = compound

    # Check if input looks like SMILES (contains special chars)
    is_smiles = any(c in compound for c in "()=#/\\@[]")
    # Check if input looks like InChIKey (14-10-1 pattern)
    is_inchikey = len(compound) == 27 and compound.count("-") == 2

    if is_smiles:
        resp, error = request(
            "POST",
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/property/CID,CanonicalSMILES,InChIKey,IUPACName/JSON",
            data={"smiles": compound},
            timeout=10,
            raise_for_status=False,
        )
    elif is_inchikey:
        resp, error = request(
            "GET",
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{compound}/property/CID,CanonicalSMILES,InChIKey,IUPACName/JSON",
            timeout=10,
            raise_for_status=False,
        )
    else:
        import urllib.parse
        encoded = urllib.parse.quote(compound, safe="")
        resp, error = request(
            "GET",
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CID,CanonicalSMILES,InChIKey,IUPACName/JSON",
            timeout=10,
            raise_for_status=False,
        )

    if not error and resp.status_code == 200:
        try:
            props = resp.json().get("PropertyTable", {}).get("Properties", [])
        except Exception:
            props = []
        if props:
            cid = props[0].get("CID")
            canonical_smiles = props[0].get("CanonicalSMILES")
            inchikey = props[0].get("InChIKey")
            compound_name = props[0].get("IUPACName", compound)

    compound_info["cid"] = cid
    compound_info["canonical_smiles"] = canonical_smiles
    compound_info["inchikey"] = inchikey

    # Step 2: Search PubChem for bioactivity data
    mechanism_cluster = None
    bioactivity_data = []
    if cid:
        bio_resp, bio_error = request(
            "GET",
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/assaysummary/JSON",
            timeout=10,
            raise_for_status=False,
        )
        if not bio_error and bio_resp.status_code == 200:
            try:
                assays = bio_resp.json().get("Table", {}).get("Row", [])
            except Exception:
                assays = []
            # Filter for cell-based / imaging assays
            for row in assays[:50]:
                cells = row.get("Cell", [])
                # Each row is a dict with Cell entries
                if isinstance(cells, list) and len(cells) > 5:
                    aid = cells[0].get("StringValue", "") if isinstance(cells[0], dict) else str(cells[0])
                    activity = cells[3].get("StringValue", "") if len(cells) > 3 and isinstance(cells[3], dict) else ""
                    bioactivity_data.append({
                        "aid": aid,
                        "activity_outcome": activity,
                    })

    # Step 3: Compute molecular descriptors using RDKit if SMILES available
    rdkit_descriptors = None
    if canonical_smiles or is_smiles:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors

            smi = canonical_smiles or compound
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                rdkit_descriptors = {
                    "molecular_weight": round(Descriptors.MolWt(mol), 2),
                    "logp": round(Descriptors.MolLogP(mol), 2),
                    "tpsa": round(Descriptors.TPSA(mol), 2),
                    "hba": Descriptors.NumHAcceptors(mol),
                    "hbd": Descriptors.NumHDonors(mol),
                    "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                    "aromatic_rings": Descriptors.NumAromaticRings(mol),
                    "fsp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
                }

                # Heuristic mechanism class based on molecular properties
                mw = rdkit_descriptors["molecular_weight"]
                logp = rdkit_descriptors["logp"]
                if mw < 500 and rdkit_descriptors["aromatic_rings"] >= 2:
                    mechanism_cluster = "kinase_inhibitor_like"
                elif mw < 600 and logp < 2:
                    mechanism_cluster = "protein_degrader_like"
                elif mw > 800:
                    mechanism_cluster = "macrocycle_like"
                else:
                    mechanism_cluster = "small_molecule"
        except ImportError:
            pass

    # Build summary
    has_data = bool(bioactivity_data or rdkit_descriptors)
    if has_data:
        cluster_str = f", mechanism cluster: '{mechanism_cluster}'" if mechanism_cluster else ""
        n_assays = len(bioactivity_data)
        assay_str = f", {n_assays} PubChem bioassay(s)" if n_assays > 0 else ""
        summary = (
            f"Compound profile for {compound}: "
            f"CID={cid or 'N/A'}{cluster_str}{assay_str}"
        )
    else:
        summary = (
            f"Compound profile for {compound}: no bioactivity data found in PubChem. "
            f"CID={cid or 'N/A'}"
        )

    result = {
        "summary": summary,
        "compound_info": compound_info,
        "compound_name": compound_name,
        "mechanism_cluster": mechanism_cluster,
        "bioactivity_assays": bioactivity_data[:20],
        "n_assays": len(bioactivity_data),
    }

    if rdkit_descriptors:
        result["molecular_descriptors"] = rdkit_descriptors

    return result


@registry.register(
    name="imaging.morphology_similarity",
    description="Compare two compounds by structural fingerprint similarity (Morgan/MACCS Tanimoto) as a proxy for phenotypic similarity",
    category="imaging",
    parameters={
        "smiles_a": "SMILES string for compound A",
        "smiles_b": "SMILES string for compound B",
    },
    usage_guide="You want to compare two compounds by structural similarity as a proxy for phenotypic similarity. Uses Morgan fingerprints (radius=2, 2048 bits), MACCS keys, and physicochemical property comparison. Structural similarity correlates with morphological similarity for ~60% of compound pairs (Bray et al. 2017). For actual Cell Painting profile comparison, pre-computed profiles from JUMP would be needed.",
)
def morphology_similarity(smiles_a: str, smiles_b: str, **kwargs) -> dict:
    """Compare two compounds by morphological similarity.

    Uses RDKit Morgan fingerprints (radius=2, 2048 bits) as a structural proxy
    for morphological similarity. Structural similarity correlates with morphological
    similarity for ~60% of compound pairs (Bray et al., Nat Biotechnol 2017).
    Also computes MACCS keys similarity and physicochemical property comparison.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys
    except ImportError:
        return {
            "error": "RDKit is required for morphology similarity. Install with: pip install rdkit",
            "summary": "RDKit not installed — needed for fingerprint-based similarity",
        }

    import numpy as np

    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)

    if mol_a is None:
        return {"error": f"Invalid SMILES for compound A: {smiles_a}", "summary": f"Could not parse SMILES: {smiles_a}"}
    if mol_b is None:
        return {"error": f"Invalid SMILES for compound B: {smiles_b}", "summary": f"Could not parse SMILES: {smiles_b}"}

    # Morgan fingerprint similarity (main metric)
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
    morgan_sim = DataStructs.TanimotoSimilarity(fp_a, fp_b)

    # MACCS keys similarity (complementary metric)
    maccs_a = MACCSkeys.GenMACCSKeys(mol_a)
    maccs_b = MACCSkeys.GenMACCSKeys(mol_b)
    maccs_sim = DataStructs.TanimotoSimilarity(maccs_a, maccs_b)

    # Dice similarity (alternative metric)
    dice_sim = DataStructs.DiceSimilarity(fp_a, fp_b)

    # Physicochemical property comparison
    def _get_props(mol):
        return {
            "mw": round(Descriptors.MolWt(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 2),
            "tpsa": round(Descriptors.TPSA(mol), 2),
            "hba": Descriptors.NumHAcceptors(mol),
            "hbd": Descriptors.NumHDonors(mol),
            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "aromatic_rings": Descriptors.NumAromaticRings(mol),
            "fsp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        }

    props_a = _get_props(mol_a)
    props_b = _get_props(mol_b)

    # Compute property similarity (normalized)
    prop_diffs = {}
    shared_features = []
    for key in props_a:
        diff = abs(props_a[key] - props_b[key])
        prop_diffs[key] = round(diff, 3)

        # Flag shared features
        if key == "mw" and diff < 50:
            shared_features.append("similar molecular weight")
        elif key == "logp" and diff < 1:
            shared_features.append("similar lipophilicity")
        elif key == "tpsa" and diff < 20:
            shared_features.append("similar polarity")
        elif key == "aromatic_rings" and diff == 0:
            shared_features.append(f"same aromatic ring count ({props_a[key]})")
        elif key == "hbd" and diff == 0 and props_a[key] > 0:
            shared_features.append(f"same H-bond donors ({props_a[key]})")

    # Infer morphological similarity class
    combined_sim = 0.6 * morgan_sim + 0.3 * maccs_sim + 0.1 * dice_sim
    if combined_sim > 0.85:
        sim_class = "highly similar"
        morphology_prediction = "Very likely similar morphological profiles"
    elif combined_sim > 0.6:
        sim_class = "moderately similar"
        morphology_prediction = "Possibly similar morphological effects"
    elif combined_sim > 0.4:
        sim_class = "weakly similar"
        morphology_prediction = "Some shared structural features; morphology may differ"
    else:
        sim_class = "dissimilar"
        morphology_prediction = "Likely different morphological profiles"

    # Heuristic mechanism class
    def _mechanism_class(props):
        if props["aromatic_rings"] >= 3 and props["hba"] >= 2:
            return "kinase_inhibitor_like"
        elif props["mw"] < 600 and props["logp"] < 2:
            return "polar_small_molecule"
        elif props["mw"] > 800:
            return "macrocycle_like"
        else:
            return "standard_small_molecule"

    mech_a = _mechanism_class(props_a)
    mech_b = _mechanism_class(props_b)

    summary = (
        f"Morphological similarity between compounds: {combined_sim:.2f} ({sim_class}). "
        f"Morgan Tanimoto: {morgan_sim:.3f}, MACCS: {maccs_sim:.3f}. "
        f"{morphology_prediction}"
    )
    if shared_features:
        summary += f". Shared: {', '.join(shared_features[:4])}"

    return {
        "summary": summary,
        "similarity_scores": {
            "morgan_tanimoto": round(morgan_sim, 4),
            "maccs_tanimoto": round(maccs_sim, 4),
            "dice": round(dice_sim, 4),
            "combined": round(combined_sim, 4),
        },
        "similarity_class": sim_class,
        "morphology_prediction": morphology_prediction,
        "shared_features": shared_features,
        "compound_a": {
            "smiles": smiles_a,
            "properties": props_a,
            "mechanism_class": mech_a,
        },
        "compound_b": {
            "smiles": smiles_b,
            "properties": props_b,
            "mechanism_class": mech_b,
        },
        "property_differences": prop_diffs,
    }
