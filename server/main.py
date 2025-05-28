import os
import pandas as pd
import requests
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import Draw
import base64
from io import BytesIO
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

from discovery import build_prompt, smiles_to_image_base64
from model import DrugTargetModel, encode_protein, fetch_drug_smiles, fetch_protein_sequence, smiles_to_fingerprint
from utils import build_predict_prompt, get_top_similar_drugs

from google import genai 

api_key = os.getenv("GEMINI_API_KEY")
ai_client = genai.Client(api_key=api_key)

protein_dim = 400  
drug_dim = 1024 
model = DrugTargetModel(protein_dim, drug_dim)
model.load_state_dict(torch.load("drug_target_model.pth"))
model.eval()

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

def generate_molecule_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base_64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_base_64}"
    return None

def compute_feature_importance(model, protein_tensor, drug_tensor):
    protein_tensor.requires_grad = True
    drug_tensor.requires_grad = True

    output = model(protein_tensor, drug_tensor)
    model.zero_grad()
    output.backward()

    protein_grad = protein_tensor.grad.abs().numpy().flatten()
    drug_grad = drug_tensor.grad.abs().numpy().flatten()

    protein_grad = (protein_grad - protein_grad.min()) / (protein_grad.max() - protein_grad.min())
    drug_grad = (drug_grad - drug_grad.min()) / (drug_grad.max() - drug_grad.min())

    return protein_grad, drug_grad

def generate_heatmap(protein_grad, drug_grad):
    interaction_matrix = np.outer(protein_grad, drug_grad)
    plt.figure(figsize=(10, 6))
    sns.heatmap(interaction_matrix, cmap="coolwarm", xticklabels=False, yticklabels=False, center=0)
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    img_buf.seek(0)
    img_str = base64.b64encode(img_buf.read()).decode("utf-8")
    plt.close()
    
    return f"data:image/png;base64,{img_str}"

app = FastAPI()

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    uniprot_id: str
    drug_name: str


@app.get("/health")
def check_health():
    return {"message": "Good Health"}

@app.post("/predict")
def predict_interaction(request: PredictionRequest):
    protein_sequence = fetch_protein_sequence(request.uniprot_id)
    if not protein_sequence:
        return {"error": "Invalid UniProt ID or sequence not found"}
    
    drug_info = fetch_drug_info(request.drug_name)
    if not drug_info or drug_info["smiles"] == "N/A":
        return {"error": "Invalid drug name or properties not found"}
    
    drug_info["name"] = request.drug_name
    
    protein_encoded = encode_protein(protein_sequence)
    drug_encoded = smiles_to_fingerprint(drug_info["smiles"])
    
    protein_tensor = torch.tensor([protein_encoded], dtype=torch.float32)
    drug_tensor = torch.tensor([drug_encoded], dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(protein_tensor, drug_tensor).item()
    
    molecule_image = generate_molecule_image(drug_info["smiles"])

    protein_grad, drug_grad = compute_feature_importance(model, protein_tensor, drug_tensor)
    heatmap_image = generate_heatmap(protein_grad, drug_grad)

    prompt = build_predict_prompt(request.uniprot_id, drug_info, prediction)
    response = ai_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    insights = response.text

    
    return {
        "uniprot_id": request.uniprot_id,
        "drug_name": request.drug_name,
        "smiles": drug_info["smiles"],
        "molecular_weight": drug_info["molecular_weight"],
        "logP": drug_info["logP"],
        "h_bond_donors": drug_info["h_bond_donors"],
        "h_bond_acceptors": drug_info["h_bond_acceptors"],
        "binding_probability": prediction,
        "molecule_image": molecule_image,
        "heatmap_image": heatmap_image,
        "top_similar_drugs": get_top_similar_drugs(drug_info['smiles']),
        "insights": insights
    }

class DiscoveryRequest(BaseModel):
    uniprot_id: str
    top_n: int = 5
    

@app.post("/discover")
def discover_candidates(request: DiscoveryRequest):
    protein_sequence = fetch_protein_sequence(request.uniprot_id)
    if not protein_sequence:
        return {"error": "Invalid UniProt ID or sequence not found"}

    protein_encoded = encode_protein(protein_sequence)
    protein_tensor = torch.tensor([protein_encoded], dtype=torch.float32)
    
    drug_db = pd.read_csv("drug_db.csv")
    results = []
    for _, row in drug_db.iterrows():
        smiles = row['smiles']
        name = row['name']

        drug_info = fetch_drug_info(name)
        if not drug_info or drug_info["smiles"] == "N/A":
            continue

        drug_fp = smiles_to_fingerprint(smiles)
        drug_tensor = torch.tensor([drug_fp], dtype=torch.float32)

        with torch.no_grad():
            score = model(protein_tensor, drug_tensor).item()

        image_base64 = smiles_to_image_base64(smiles)

        results.append({
            "name": name,
            "smiles": smiles,
            "score": round(score, 4),
            "image_base64": image_base64,
            **drug_info
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:request.top_n]

    prompt = build_prompt(request.uniprot_id, results)
    response = ai_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    insights = response.text

    return {
        "uniprot_id": request.uniprot_id,
        "top_candidates": results,
        "insights": insights
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
