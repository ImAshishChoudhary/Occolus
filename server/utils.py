from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import pandas as pd
from rdkit.Chem import Draw
from io import BytesIO
import base64

def tanimoto_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def smiles_to_image_base64(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(200, 200))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img64}"

def get_top_similar_drugs(query_smiles, drug_db_path='drug_db.csv', top_n=5):
    df = pd.read_csv(drug_db_path)
    similarities = []

    for _, row in df.iterrows():
        sim = tanimoto_similarity(query_smiles, row['smiles'])
        img_base64 = smiles_to_image_base64(row['smiles'])
        similarities.append({
            "name": row['name'],
            "smiles": row['smiles'],
            "similarity": round(sim, 3),
            "image_base64": img_base64
        })

    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_n]


def build_predict_prompt(uniprot_id, drug_info, binding_score):
    return f"""
The following is the result of a drug-target interaction prediction:

- Target Protein (UniProt ID): {uniprot_id}
- Drug Name: {drug_info['name']}
- SMILES: {drug_info['smiles']}
- Molecular Weight: {drug_info['molecular_weight']}
- LogP: {drug_info['logP']}
- H-bond Donors: {drug_info['h_bond_donors']}
- H-bond Acceptors: {drug_info['h_bond_acceptors']}
- Predicted Binding Score (0–1): {binding_score:.4f}

Can you provide:
1. An interpretation of the binding score
2. Whether the drug appears drug-like
3. Any known uses or potential repurposing
4. Suggestions for next steps if this was a lead compound
"""
