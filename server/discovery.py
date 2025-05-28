import os
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import numpy as np
import pandas as pd
import requests
from io import BytesIO
import base64


def fetch_protein_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        lines = response.text.split("\n")
        return "".join(lines[1:])
    return None

def fetch_drug_info(drug_name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/CanonicalSMILES,MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount/JSON"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        properties = data.get("PropertyTable", {}).get("Properties", [{}])[0]
        return {
            "smiles": properties.get("CanonicalSMILES", "N/A"),
            "molecular_weight": properties.get("MolecularWeight", "N/A"),
            "logP": properties.get("XLogP", "N/A"),
            "h_bond_donors": properties.get("HBondDonorCount", "N/A"),
            "h_bond_acceptors": properties.get("HBondAcceptorCount", "N/A")
        }
    return None

def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

def encode_protein(protein, max_len=20):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    encoding = np.zeros((max_len, len(amino_acids)))
    for i, aa in enumerate(protein[:max_len]):
        if aa in amino_acids:
            encoding[i, amino_acids.index(aa)] = 1
    return encoding.flatten()

def smiles_to_image_base64(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(200, 200))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def build_prompt(protein_name, top_candidates):
    prompt = f"Target protein: {protein_name}\nTop predicted binders:\n\n"
    for i, drug in enumerate(top_candidates, 1):
        prompt += (
            f"{i}. {drug['name']} - Score: {drug['score']}, "
            f"MW: {drug.get('molecular_weight', 'N/A')}, "
            f"LogP: {drug.get('logP', 'N/A')}, "
            f"HBond Donors: {drug.get('h_bond_donors', 'N/A')}, "
            f"HBond Acceptors: {drug.get('h_bond_acceptors', 'N/A')}\n"
        )
    prompt += (
        "\nPlease analyze these results. Are these compounds drug-like? "
        "Any known use, literature relevance, or repurposing opportunities?\n"
        "\nPlease return results in Markdown and please dont send tabulated results.\n"
    )
    print(prompt)
    return prompt
