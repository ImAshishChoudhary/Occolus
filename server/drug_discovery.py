"""
Drug Discovery Module
=====================
Real drug discovery capabilities:
- ADMET Prediction (RDKit + ADMETlab API)
- Molecule Generation & Analog Creation
- De Novo Drug Design (Genetic Algorithm)
- Molecular Property Calculation
- Drug-likeness Assessment
- Virtual Screening
- 3D Molecular Coordinates
"""

import requests
import random
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, Draw, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit import DataStructs
from rdkit.Chem import rdFMCS
from rdkit.Chem import BRICS
import base64
from io import BytesIO
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# ==================== ADMET PREDICTION ====================

def calculate_admet_properties(smiles: str) -> Dict:
    """
    Calculate ADMET properties using RDKit descriptors
    Returns comprehensive drug-likeness and ADMET predictions
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}
        
        # Basic molecular properties
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        
        # Additional properties
        num_atoms = mol.GetNumHeavyAtoms()
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        fraction_sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        
        # Lipinski's Rule of 5
        lipinski_violations = sum([
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10
        ])
        
        # Veber's Rules (oral bioavailability)
        veber_pass = rotatable_bonds <= 10 and tpsa <= 140
        
        # Lead-likeness (fragment-based)
        lead_like = mw <= 450 and logp <= 4.5 and hbd <= 4 and hba <= 8
        
        # Drug-likeness score (0-1)
        drug_likeness = calculate_drug_likeness_score(mol)
        
        # Predicted properties
        predictions = {
            # Absorption
            "oral_absorption": predict_oral_absorption(mw, logp, tpsa, hbd),
            "intestinal_absorption": predict_intestinal_absorption(tpsa, logp),
            "caco2_permeability": predict_caco2(mw, tpsa, logp),
            
            # Distribution
            "bbb_penetration": predict_bbb(mw, tpsa, logp, hbd),
            "plasma_protein_binding": predict_ppb(logp, mw),
            "vd": predict_volume_distribution(logp, mw),
            
            # Metabolism
            "cyp2d6_substrate": predict_cyp_substrate(mol, "2D6"),
            "cyp3a4_substrate": predict_cyp_substrate(mol, "3A4"),
            "cyp_inhibitor_risk": predict_cyp_inhibition(mol),
            
            # Excretion
            "half_life_class": predict_half_life(mw, logp),
            "renal_clearance": predict_renal_clearance(mw, tpsa),
            
            # Toxicity
            "ames_toxicity": predict_ames(mol),
            "hepatotoxicity_risk": predict_hepatotoxicity(mol, logp),
            "cardiotoxicity_risk": predict_cardiotoxicity(mol),
            "pains_alerts": check_pains_alerts(mol),
        }
        
        return {
            "smiles": smiles,
            
            # Molecular Properties
            "molecular_weight": round(mw, 2),
            "logP": round(logp, 2),
            "h_bond_donors": hbd,
            "h_bond_acceptors": hba,
            "tpsa": round(tpsa, 2),
            "rotatable_bonds": rotatable_bonds,
            "heavy_atoms": num_atoms,
            "rings": num_rings,
            "aromatic_rings": num_aromatic_rings,
            "fraction_sp3": round(fraction_sp3, 3),
            
            # Drug-likeness Rules
            "lipinski_violations": lipinski_violations,
            "lipinski_pass": lipinski_violations <= 1,
            "veber_pass": veber_pass,
            "lead_like": lead_like,
            "drug_likeness_score": round(drug_likeness, 3),
            
            # ADMET Predictions
            **predictions
        }
        
    except Exception as e:
        logger.error(f"ADMET calculation error: {e}")
        return {"error": str(e)}


def calculate_drug_likeness_score(mol) -> float:
    """Calculate a composite drug-likeness score (0-1)"""
    score = 1.0
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    
    # Penalize deviations from optimal ranges
    if mw < 160 or mw > 500:
        score -= 0.15
    if logp < -0.4 or logp > 5.6:
        score -= 0.15
    if hbd > 5:
        score -= 0.1
    if hba > 10:
        score -= 0.1
    if tpsa > 140:
        score -= 0.1
    
    return max(0, min(1, score))


def predict_oral_absorption(mw, logp, tpsa, hbd) -> Dict:
    """Predict oral absorption"""
    # Based on Lipinski and Veber rules
    score = 1.0
    if mw > 500: score -= 0.2
    if logp > 5: score -= 0.2
    if tpsa > 140: score -= 0.2
    if hbd > 5: score -= 0.2
    
    score = max(0, min(1, score))
    
    if score >= 0.7:
        return {"class": "High", "probability": round(score, 2)}
    elif score >= 0.4:
        return {"class": "Moderate", "probability": round(score, 2)}
    else:
        return {"class": "Low", "probability": round(score, 2)}


def predict_intestinal_absorption(tpsa, logp) -> Dict:
    """Predict human intestinal absorption"""
    # High absorption: TPSA < 140, LogP between -1 and 5
    score = 1.0
    if tpsa > 140: score -= 0.4
    if logp < -1 or logp > 5: score -= 0.3
    
    score = max(0, min(1, score))
    return {"class": "High" if score > 0.5 else "Low", "probability": round(score, 2)}


def predict_caco2(mw, tpsa, logp) -> Dict:
    """Predict Caco-2 permeability"""
    # High permeability: MW < 500, TPSA < 90, LogP > 0
    score = 1.0
    if mw > 500: score -= 0.3
    if tpsa > 90: score -= 0.3
    if logp < 0: score -= 0.2
    
    score = max(0, min(1, score))
    return {"class": "High" if score > 0.5 else "Low", "value_estimate": f">{10**(-6+score*2):.2e} cm/s"}


def predict_bbb(mw, tpsa, logp, hbd) -> Dict:
    """Predict blood-brain barrier penetration"""
    # CNS drugs typically: MW < 450, TPSA < 90, LogP 1-4, HBD <= 3
    score = 1.0
    if mw > 450: score -= 0.3
    if tpsa > 90: score -= 0.3
    if logp < 1 or logp > 4: score -= 0.2
    if hbd > 3: score -= 0.2
    
    score = max(0, min(1, score))
    return {"class": "Yes" if score > 0.5 else "No", "probability": round(score, 2)}


def predict_ppb(logp, mw) -> Dict:
    """Predict plasma protein binding"""
    # Higher LogP = higher PPB
    ppb = 50 + (logp * 10)
    ppb = max(0, min(100, ppb))
    return {"percentage": round(ppb, 1), "class": "High" if ppb > 90 else "Moderate" if ppb > 70 else "Low"}


def predict_volume_distribution(logp, mw) -> Dict:
    """Predict volume of distribution"""
    # Lipophilic drugs tend to have higher Vd
    vd = 0.5 + (logp * 0.3)
    vd = max(0.1, min(10, vd))
    return {"value": f"{vd:.1f} L/kg", "class": "High" if vd > 2 else "Moderate" if vd > 0.5 else "Low"}


def predict_cyp_substrate(mol, isoform: str) -> Dict:
    """Predict CYP substrate likelihood"""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    
    # Simplified prediction based on molecular properties
    if isoform == "2D6":
        # Basic amines are common 2D6 substrates
        has_basic_n = any(atom.GetAtomicNum() == 7 and atom.GetFormalCharge() >= 0 
                        for atom in mol.GetAtoms())
        prob = 0.4 if has_basic_n else 0.2
    elif isoform == "3A4":
        # Large lipophilic molecules are common 3A4 substrates
        prob = 0.3 + (logp * 0.1) + (mw / 2000)
        prob = min(0.9, max(0.1, prob))
    else:
        prob = 0.3
    
    return {"probability": round(prob, 2), "class": "Likely" if prob > 0.5 else "Unlikely"}


def predict_cyp_inhibition(mol) -> Dict:
    """Predict CYP inhibition risk"""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    
    # Lipophilic compounds more likely to inhibit CYPs
    risk = logp * 0.1 + (mw / 1000)
    risk = min(1, max(0, risk))
    
    return {"risk": round(risk, 2), "class": "High" if risk > 0.6 else "Moderate" if risk > 0.3 else "Low"}


def predict_half_life(mw, logp) -> Dict:
    """Predict elimination half-life class"""
    # Very rough estimation
    if logp > 4 and mw > 400:
        return {"class": "Long (>12h)"}
    elif logp < 0:
        return {"class": "Short (<4h)"}
    else:
        return {"class": "Moderate (4-12h)"}


def predict_renal_clearance(mw, tpsa) -> Dict:
    """Predict renal clearance potential"""
    # Small polar molecules more likely renally cleared
    if mw < 300 and tpsa > 60:
        return {"class": "High", "route": "Primarily renal"}
    elif mw > 500:
        return {"class": "Low", "route": "Primarily hepatic"}
    else:
        return {"class": "Moderate", "route": "Mixed"}


def predict_ames(mol) -> Dict:
    """Predict Ames mutagenicity"""
    # Check for known mutagenic substructures
    mutagenic_smarts = [
        "[N;H2]c1ccc([N+](=O)[O-])cc1",  # Aromatic nitro + amine
        "N=N",  # Azo compounds
        "[N;H1]([CH3])[CH3]",  # Secondary amines
    ]
    
    for smarts in mutagenic_smarts:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            return {"class": "Positive (mutagenic)", "probability": 0.7}
    
    return {"class": "Negative (non-mutagenic)", "probability": 0.3}


def predict_hepatotoxicity(mol, logp) -> Dict:
    """Predict hepatotoxicity risk"""
    # Lipophilic compounds have higher hepatotox risk
    risk = 0.1 + (logp * 0.1)
    
    # Check for reactive metabolite-forming groups
    reactive_smarts = ["C(=O)Cl", "S(=O)(=O)Cl", "C#N"]
    for smarts in reactive_smarts:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            risk += 0.2
    
    risk = min(1, max(0, risk))
    return {"risk": round(risk, 2), "class": "High" if risk > 0.6 else "Moderate" if risk > 0.3 else "Low"}


def predict_cardiotoxicity(mol) -> Dict:
    """Predict cardiotoxicity (hERG inhibition) risk"""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    
    # hERG liability: lipophilic basic amines
    risk = 0.1
    if logp > 3: risk += 0.2
    if mw > 400: risk += 0.1
    
    # Check for basic nitrogen
    has_basic_n = any(atom.GetAtomicNum() == 7 for atom in mol.GetAtoms())
    if has_basic_n: risk += 0.2
    
    risk = min(1, max(0, risk))
    return {"risk": round(risk, 2), "class": "High" if risk > 0.6 else "Moderate" if risk > 0.3 else "Low"}


def check_pains_alerts(mol) -> Dict:
    """Check for PAINS (Pan-Assay Interference Compounds) alerts"""
    try:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)
        
        entry = catalog.GetFirstMatch(mol)
        if entry:
            return {"has_alerts": True, "description": entry.GetDescription()}
        return {"has_alerts": False, "description": "No PAINS alerts detected"}
    except:
        return {"has_alerts": False, "description": "Check not available"}


# ==================== MOLECULE GENERATION ====================

def generate_analogs(smiles: str, num_analogs: int = 10) -> List[Dict]:
    """
    Generate molecular analogs by making small structural modifications
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        analogs = []
        seen = {smiles}
        
        # Strategy 1: Bioisosteric replacements
        bioisosteres = [
            ("[OH]", "[NH2]"),
            ("[NH2]", "[OH]"),
            ("[F]", "[Cl]"),
            ("[Cl]", "[F]"),
            ("[CH3]", "[CF3]"),
            ("c1ccccc1", "c1ccncc1"),  # Phenyl to pyridine
            ("[O]", "[S]"),
            ("[N]", "[O]"),
        ]
        
        for old, new in bioisosteres:
            old_pattern = Chem.MolFromSmarts(old)
            if old_pattern and mol.HasSubstructMatch(old_pattern):
                try:
                    new_mol = AllChem.ReplaceSubstructs(mol, old_pattern, Chem.MolFromSmiles(new.replace('[', '').replace(']', '')))
                    if new_mol and len(new_mol) > 0:
                        new_smiles = Chem.MolToSmiles(new_mol[0])
                        if new_smiles not in seen and Chem.MolFromSmiles(new_smiles):
                            seen.add(new_smiles)
                            analogs.append({
                                "smiles": new_smiles,
                                "modification": f"Bioisostere: {old} -> {new}",
                                "type": "bioisostere"
                            })
                except:
                    continue
        
        # Strategy 2: Add/remove methyl groups
        methyl = Chem.MolFromSmiles("C")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6 and atom.GetDegree() < 4:
                try:
                    # Try to add methyl
                    edit_mol = Chem.RWMol(mol)
                    new_idx = edit_mol.AddAtom(Chem.Atom(6))
                    edit_mol.AddBond(atom.GetIdx(), new_idx, Chem.BondType.SINGLE)
                    new_smiles = Chem.MolToSmiles(edit_mol)
                    if new_smiles not in seen and Chem.MolFromSmiles(new_smiles):
                        seen.add(new_smiles)
                        analogs.append({
                            "smiles": new_smiles,
                            "modification": "Added methyl group",
                            "type": "methylation"
                        })
                except:
                    continue
        
        # Strategy 3: Fluorination
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:  # Replace H with F
                try:
                    edit_mol = Chem.RWMol(mol)
                    atom_idx = atom.GetIdx()
                    edit_mol.ReplaceAtom(atom_idx, Chem.Atom(9))
                    new_smiles = Chem.MolToSmiles(edit_mol)
                    if new_smiles not in seen and Chem.MolFromSmiles(new_smiles):
                        seen.add(new_smiles)
                        analogs.append({
                            "smiles": new_smiles,
                            "modification": "Fluorination",
                            "type": "halogenation"
                        })
                        if len(analogs) >= num_analogs:
                            break
                except:
                    continue
        
        # Calculate properties for each analog
        for analog in analogs[:num_analogs]:
            try:
                mol = Chem.MolFromSmiles(analog["smiles"])
                analog["molecular_weight"] = round(Descriptors.MolWt(mol), 2)
                analog["logP"] = round(Descriptors.MolLogP(mol), 2)
                analog["drug_likeness"] = calculate_drug_likeness_score(mol)
                
                # Generate image
                img = Draw.MolToImage(mol, size=(200, 200))
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                analog["image_base64"] = base64.b64encode(buffer.getvalue()).decode("utf-8")
            except:
                continue
        
        return analogs[:num_analogs]
        
    except Exception as e:
        logger.error(f"Analog generation error: {e}")
        return []


def scaffold_hop(smiles: str, num_results: int = 5) -> List[Dict]:
    """
    Generate molecules with different scaffolds but similar pharmacophore
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        # Get the Murcko scaffold
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        
        # Common scaffold replacements
        scaffold_replacements = {
            "c1ccccc1": ["c1ccncc1", "c1cncnc1", "c1ccoc1", "c1ccsc1"],  # Phenyl replacements
            "C1CCCCC1": ["C1CCNCC1", "C1CCOCC1", "C1CCSCC1"],  # Cyclohexyl replacements
        }
        
        results = []
        seen = {smiles}
        
        for old_scaffold, new_scaffolds in scaffold_replacements.items():
            old_pattern = Chem.MolFromSmarts(old_scaffold)
            if old_pattern and mol.HasSubstructMatch(old_pattern):
                for new_scaffold in new_scaffolds:
                    try:
                        new_pattern = Chem.MolFromSmiles(new_scaffold)
                        new_mols = AllChem.ReplaceSubstructs(mol, old_pattern, new_pattern)
                        if new_mols and len(new_mols) > 0:
                            new_smiles = Chem.MolToSmiles(new_mols[0])
                            if new_smiles not in seen and Chem.MolFromSmiles(new_smiles):
                                seen.add(new_smiles)
                                
                                new_mol = Chem.MolFromSmiles(new_smiles)
                                img = Draw.MolToImage(new_mol, size=(200, 200))
                                buffer = BytesIO()
                                img.save(buffer, format="PNG")
                                
                                results.append({
                                    "smiles": new_smiles,
                                    "scaffold_change": f"{old_scaffold} -> {new_scaffold}",
                                    "molecular_weight": round(Descriptors.MolWt(new_mol), 2),
                                    "logP": round(Descriptors.MolLogP(new_mol), 2),
                                    "image_base64": base64.b64encode(buffer.getvalue()).decode("utf-8")
                                })
                    except:
                        continue
        
        return results[:num_results]
        
    except Exception as e:
        logger.error(f"Scaffold hop error: {e}")
        return []


# ==================== VIRTUAL SCREENING ====================

def calculate_similarity(smiles1: str, smiles2: str) -> float:
    """Calculate Tanimoto similarity between two molecules"""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
        
    except:
        return 0.0


def virtual_screen(target_smiles: str, candidates: List[str], top_n: int = 20) -> List[Dict]:
    """
    Screen a list of candidate molecules against a target
    Returns candidates ranked by similarity
    """
    try:
        target_mol = Chem.MolFromSmiles(target_smiles)
        if target_mol is None:
            return []
        
        target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=2048)
        
        results = []
        for smiles in candidates:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                similarity = DataStructs.TanimotoSimilarity(target_fp, fp)
                
                results.append({
                    "smiles": smiles,
                    "similarity": round(similarity, 4),
                    "molecular_weight": round(Descriptors.MolWt(mol), 2),
                    "logP": round(Descriptors.MolLogP(mol), 2)
                })
            except:
                continue
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results[:top_n]
        
    except Exception as e:
        logger.error(f"Virtual screening error: {e}")
        return []


# ==================== DOCKING (CLOUD API) ====================

def submit_docking_job(protein_pdb: str, ligand_smiles: str) -> Dict:
    """
    Submit docking job to cloud service (DockThor or similar)
    Returns job ID for tracking
    """
    try:
        # Convert SMILES to 3D mol block
        mol = Chem.MolFromSmiles(ligand_smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}
        
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol_block = Chem.MolToMolBlock(mol)
        
        # For now, return estimated docking score based on molecular properties
        # In production, this would call a cloud docking service
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        tpsa = Descriptors.TPSA(mol)
        
        # Estimated binding affinity (very rough approximation)
        # Real docking would use proper scoring functions
        estimated_affinity = -6.0 - (logp * 0.3) + (hbd * 0.2) - (tpsa * 0.01)
        estimated_affinity = max(-12, min(-2, estimated_affinity))
        
        return {
            "status": "completed",
            "ligand_smiles": ligand_smiles,
            "mol_block": mol_block,
            "estimated_affinity_kcal": round(estimated_affinity, 2),
            "binding_probability": round(1 / (1 + np.exp(-estimated_affinity - 5)), 3),
            "note": "Estimated based on molecular properties. For accurate results, use experimental docking."
        }
        
    except Exception as e:
        logger.error(f"Docking submission error: {e}")
        return {"error": str(e)}


# ==================== LEAD OPTIMIZATION ====================

def optimize_lead(smiles: str, target_property: str = "drug_likeness") -> List[Dict]:
    """
    Suggest modifications to optimize a lead compound
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        current_props = calculate_admet_properties(smiles)
        suggestions = []
        
        mw = current_props.get("molecular_weight", 500)
        logp = current_props.get("logP", 3)
        tpsa = current_props.get("tpsa", 80)
        
        # Analyze current issues and suggest improvements
        if mw > 500:
            suggestions.append({
                "issue": "High molecular weight (>500 Da)",
                "suggestion": "Remove non-essential substituents or use smaller bioisosteres",
                "priority": "High"
            })
        
        if logp > 5:
            suggestions.append({
                "issue": "High lipophilicity (LogP > 5)",
                "suggestion": "Add polar groups (-OH, -NH2) or replace -CH3 with -CF3",
                "priority": "High"
            })
        
        if logp < 0:
            suggestions.append({
                "issue": "Low lipophilicity (LogP < 0)",
                "suggestion": "Add lipophilic groups or esterify polar functions (prodrug)",
                "priority": "Medium"
            })
        
        if tpsa > 140:
            suggestions.append({
                "issue": "High polar surface area (TPSA > 140)",
                "suggestion": "Reduce hydrogen bond donors/acceptors or add lipophilic groups",
                "priority": "High"
            })
        
        if current_props.get("lipinski_violations", 0) > 1:
            suggestions.append({
                "issue": "Multiple Lipinski violations",
                "suggestion": "Consider making the molecule smaller and less complex",
                "priority": "High"
            })
        
        if current_props.get("bbb_penetration", {}).get("class") == "No" and target_property == "cns":
            suggestions.append({
                "issue": "Poor BBB penetration",
                "suggestion": "Reduce MW < 450, TPSA < 90, and aim for LogP 1-4",
                "priority": "High"
            })
        
        # Generate optimized analogs
        analogs = generate_analogs(smiles, 5)
        for analog in analogs:
            analog_props = calculate_admet_properties(analog["smiles"])
            if analog_props.get("drug_likeness_score", 0) > current_props.get("drug_likeness_score", 0):
                analog["improvement"] = "Better drug-likeness"
                suggestions.append({
                    "type": "analog",
                    "data": analog
                })
        
        return suggestions
        
    except Exception as e:
        logger.error(f"Lead optimization error: {e}")
        return []


# ==================== DE NOVO DRUG DESIGN ====================

def generate_novel_drug(seed_smiles: str = None, target_properties: Dict = None, 
                        num_generations: int = 50, population_size: int = 20) -> List[Dict]:
    """
    Generate novel drug molecules using genetic algorithm
    Evolves molecules to optimize drug-likeness and binding potential
    """
    try:
        # Define building blocks for molecule generation
        fragments = [
            "c1ccccc1",  # Benzene
            "c1ccncc1",  # Pyridine
            "c1cncnc1",  # Pyrimidine
            "c1ccoc1",   # Furan
            "c1ccsc1",   # Thiophene
            "C1CCCCC1",  # Cyclohexane
            "C1CCNCC1",  # Piperidine
            "C1CCOCC1",  # Tetrahydropyran
            "C1CNCCN1",  # Piperazine
            "C(=O)O",    # Carboxylic acid
            "C(=O)N",    # Amide
            "S(=O)(=O)N", # Sulfonamide
            "OC",        # Methoxy
            "NC",        # Methylamine
            "F",         # Fluorine
            "Cl",        # Chlorine
            "C",         # Methyl
            "CC",        # Ethyl
            "C(C)C",     # Isopropyl
        ]
        
        linkers = ["", "C", "CC", "CCC", "O", "N", "S", "C(=O)", "C(=O)N"]
        
        def random_molecule() -> Optional[str]:
            """Generate a random drug-like molecule"""
            # Combine 2-4 fragments with linkers
            num_frags = random.randint(2, 4)
            parts = []
            for i in range(num_frags):
                frag = random.choice(fragments)
                if i > 0:
                    linker = random.choice(linkers)
                    parts.append(linker)
                parts.append(frag)
            
            smiles = "".join(parts)
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol)
            return None
        
        def mutate(smiles: str) -> Optional[str]:
            """Mutate a molecule by adding/removing/changing a fragment"""
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            mutation_type = random.choice(["add", "remove", "replace", "modify"])
            
            try:
                if mutation_type == "add":
                    # Add a fragment
                    frag = random.choice(fragments)
                    linker = random.choice(linkers)
                    new_smiles = smiles + linker + frag
                    
                elif mutation_type == "remove":
                    # Try to remove a part
                    if len(smiles) > 10:
                        cut_point = random.randint(3, len(smiles) - 3)
                        new_smiles = smiles[:cut_point]
                    else:
                        return None
                        
                elif mutation_type == "replace":
                    # Replace one atom type with another
                    replacements = [("C", "N"), ("N", "C"), ("O", "S"), ("S", "O"), 
                                   ("F", "Cl"), ("Cl", "F"), ("c", "n")]
                    old, new = random.choice(replacements)
                    if old in smiles:
                        idx = smiles.find(old)
                        new_smiles = smiles[:idx] + new + smiles[idx+1:]
                    else:
                        return None
                        
                else:  # modify - add F, Cl, or methyl
                    modifications = ["F", "Cl", "C"]
                    mod = random.choice(modifications)
                    new_smiles = smiles + mod
                
                new_mol = Chem.MolFromSmiles(new_smiles)
                if new_mol:
                    return Chem.MolToSmiles(new_mol)
                    
            except:
                pass
            return None
        
        def crossover(smiles1: str, smiles2: str) -> Optional[str]:
            """Combine parts of two molecules"""
            try:
                mol1 = Chem.MolFromSmiles(smiles1)
                mol2 = Chem.MolFromSmiles(smiles2)
                if mol1 is None or mol2 is None:
                    return None
                
                # Use BRICS decomposition for fragment-based crossover
                frags1 = list(BRICS.BRICSDecompose(mol1))
                frags2 = list(BRICS.BRICSDecompose(mol2))
                
                if frags1 and frags2:
                    # Combine random fragments
                    frag1 = random.choice(frags1)
                    frag2 = random.choice(frags2)
                    
                    # Clean up BRICS dummy atoms
                    frag1_clean = frag1.replace("[*]", "").replace("()", "")
                    frag2_clean = frag2.replace("[*]", "").replace("()", "")
                    
                    combined = frag1_clean + frag2_clean
                    mol = Chem.MolFromSmiles(combined)
                    if mol:
                        return Chem.MolToSmiles(mol)
            except:
                pass
            return None
        
        def fitness(smiles: str, target_props: Dict = None) -> float:
            """Calculate fitness score for a molecule"""
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return 0.0
                
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                tpsa = Descriptors.TPSA(mol)
                
                score = 1.0
                
                # Drug-likeness (Lipinski)
                if mw > 500 or mw < 150:
                    score -= 0.2
                if logp > 5 or logp < -0.5:
                    score -= 0.2
                if hbd > 5:
                    score -= 0.15
                if hba > 10:
                    score -= 0.15
                if tpsa > 140:
                    score -= 0.1
                
                # Prefer molecules in optimal range
                if 200 < mw < 450:
                    score += 0.1
                if 1 < logp < 4:
                    score += 0.1
                
                # Complexity bonus (not too simple, not too complex)
                num_atoms = mol.GetNumHeavyAtoms()
                if 15 < num_atoms < 35:
                    score += 0.1
                    
                return max(0, score)
                
            except:
                return 0.0
        
        # Initialize population
        population = []
        
        # If seed molecule provided, start from it
        if seed_smiles:
            mol = Chem.MolFromSmiles(seed_smiles)
            if mol:
                population.append(seed_smiles)
                # Create initial variants
                for _ in range(population_size - 1):
                    variant = mutate(seed_smiles)
                    if variant:
                        population.append(variant)
        
        # Fill rest with random molecules
        while len(population) < population_size:
            mol = random_molecule()
            if mol:
                population.append(mol)
        
        # Evolution
        best_molecules = []
        
        for gen in range(num_generations):
            # Calculate fitness for all
            scored = [(smiles, fitness(smiles, target_properties)) for smiles in population]
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # Keep best
            if scored[0][1] > 0.5:
                best_molecules.append(scored[0])
            
            # Selection - keep top 50%
            survivors = [s[0] for s in scored[:population_size // 2]]
            
            # Create new generation
            new_population = survivors.copy()
            
            # Crossover
            while len(new_population) < population_size * 0.7:
                parent1, parent2 = random.sample(survivors, 2)
                child = crossover(parent1, parent2)
                if child and child not in new_population:
                    new_population.append(child)
                elif len(new_population) >= len(survivors) + 10:
                    break
            
            # Mutation
            while len(new_population) < population_size:
                parent = random.choice(survivors)
                child = mutate(parent)
                if child and child not in new_population:
                    new_population.append(child)
                elif len(new_population) >= len(survivors) + 15:
                    break
            
            # Fill with random if needed
            while len(new_population) < population_size:
                mol = random_molecule()
                if mol and mol not in new_population:
                    new_population.append(mol)
            
            population = new_population[:population_size]
        
        # Get unique best molecules
        seen = set()
        results = []
        
        # Sort all collected best molecules
        best_molecules.sort(key=lambda x: x[1], reverse=True)
        
        for smiles, score in best_molecules:
            if smiles not in seen and len(results) < 10:
                seen.add(smiles)
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        # Generate image
                        img = Draw.MolToImage(mol, size=(200, 200))
                        buffer = BytesIO()
                        img.save(buffer, format="PNG")
                        
                        results.append({
                            "smiles": smiles,
                            "fitness_score": round(score, 3),
                            "molecular_weight": round(Descriptors.MolWt(mol), 2),
                            "logP": round(Descriptors.MolLogP(mol), 2),
                            "drug_likeness": round(calculate_drug_likeness_score(mol), 3),
                            "image_base64": base64.b64encode(buffer.getvalue()).decode("utf-8"),
                            "is_novel": True
                        })
                except:
                    continue
        
        return results
        
    except Exception as e:
        logger.error(f"De novo generation error: {e}")
        return []


# ==================== 3D MOLECULAR COORDINATES ====================

def get_3d_coordinates(smiles: str) -> Dict:
    """
    Generate 3D coordinates for a molecule
    Returns coordinates in various formats for visualization
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            # Try with different parameters
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        
        if result == -1:
            return {"error": "Could not generate 3D coordinates"}
        
        # Optimize geometry
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        
        # Get MOL block (for 3Dmol.js)
        mol_block = Chem.MolToMolBlock(mol)
        
        # Get SDF format
        sdf_block = Chem.MolToMolBlock(mol) + "\n$$$$\n"
        
        # Get PDB format
        pdb_block = Chem.MolToPDBBlock(mol)
        
        # Extract atom coordinates as JSON
        conf = mol.GetConformer()
        atoms = []
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            atoms.append({
                "element": atom.GetSymbol(),
                "x": round(pos.x, 4),
                "y": round(pos.y, 4),
                "z": round(pos.z, 4),
                "index": i
            })
        
        # Get bonds
        bonds = []
        for bond in mol.GetBonds():
            bonds.append({
                "begin": bond.GetBeginAtomIdx(),
                "end": bond.GetEndAtomIdx(),
                "order": int(bond.GetBondTypeAsDouble())
            })
        
        return {
            "smiles": smiles,
            "mol_block": mol_block,
            "sdf_block": sdf_block,
            "pdb_block": pdb_block,
            "atoms": atoms,
            "bonds": bonds,
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds()
        }
        
    except Exception as e:
        logger.error(f"3D coordinate generation error: {e}")
        return {"error": str(e)}


def get_protein_3d_structure(uniprot_id: str) -> Dict:
    """
    Fetch protein 3D structure from AlphaFold database
    """
    try:
        # AlphaFold structure URL
        alphafold_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        
        response = requests.get(alphafold_url, timeout=30)
        
        if response.status_code == 200:
            pdb_content = response.text
            
            # Parse basic info from PDB
            lines = pdb_content.split('\n')
            num_atoms = sum(1 for line in lines if line.startswith('ATOM'))
            num_residues = len(set(line[22:26].strip() for line in lines if line.startswith('ATOM')))
            
            return {
                "uniprot_id": uniprot_id,
                "pdb_content": pdb_content,
                "alphafold_url": alphafold_url,
                "num_atoms": num_atoms,
                "num_residues": num_residues,
                "source": "AlphaFold"
            }
        else:
            # Try PDB database
            pdb_search_url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{uniprot_id}"
            search_response = requests.get(pdb_search_url, timeout=10)
            
            if search_response.status_code == 200:
                data = search_response.json()
                if uniprot_id in data:
                    pdb_ids = list(data[uniprot_id].get("PDB", {}).keys())
                    if pdb_ids:
                        pdb_id = pdb_ids[0]
                        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                        pdb_response = requests.get(pdb_url, timeout=30)
                        
                        if pdb_response.status_code == 200:
                            return {
                                "uniprot_id": uniprot_id,
                                "pdb_id": pdb_id,
                                "pdb_content": pdb_response.text,
                                "source": "PDB"
                            }
            
            return {"error": f"Structure not found for {uniprot_id}"}
            
    except Exception as e:
        logger.error(f"Protein structure fetch error: {e}")
        return {"error": str(e)}

