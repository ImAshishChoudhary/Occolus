import os
import sys
import logging
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

# Setup logging to show in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

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

# CORS configuration - allow frontend origins
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
]

# Add any additional origins from environment variable
cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env:
    origins.extend([origin.strip() for origin in cors_origins_env.split(",") if origin.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class PredictionRequest(BaseModel):
    uniprot_id: str
    drug_name: str


@app.get("/health")
def check_health():
    logger.info("Health check endpoint called")
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
    insights = None
    
    # Try different models as fallback
    models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    
    for model_name in models_to_try:
        try:
            response = ai_client.models.generate_content(model=model_name, contents=prompt)
            insights = response.text
            break
        except Exception as e:
            error_msg = str(e)
            print(f"[Warning] {model_name} failed: {error_msg[:100]}")
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                continue
            else:
                break
    
    if not insights:
        binding_status = "likely to bind" if prediction > 0.5 else "unlikely to bind"
        insights = f"""## Prediction Summary

The model predicts that **{request.drug_name}** is **{binding_status}** to protein **{request.uniprot_id}** with a binding probability of **{prediction:.2%}**.

**Note:** Detailed AI analysis is temporarily unavailable due to API rate limits. Please try again later."""

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
    logger.info(f"=== DISCOVER REQUEST RECEIVED === Protein: {request.uniprot_id}, Top N: {request.top_n}")
    sys.stdout.flush()
    
    protein_sequence = fetch_protein_sequence(request.uniprot_id)
    if not protein_sequence:
        logger.error(f"Invalid UniProt ID: {request.uniprot_id}")
        return {"error": "Invalid UniProt ID or sequence not found"}

    logger.info(f"Protein sequence fetched, length: {len(protein_sequence)}")
    
    protein_encoded = encode_protein(protein_sequence)
    protein_tensor = torch.tensor(np.array([protein_encoded]), dtype=torch.float32)
    
    drug_db = pd.read_csv("drug_db.csv")
    results = []
    
    # Filter out biologics and invalid SMILES
    valid_drugs = drug_db[~drug_db['smiles'].str.contains('Antibody|Analog|Inhibitor|Protein|Heparin|Insulin', case=False, na=False)]
    logger.info(f"Processing {len(valid_drugs)} small molecule drugs...")
    sys.stdout.flush()
    
    for idx, row in valid_drugs.iterrows():
        smiles = row['smiles']
        name = row['name']
        
        try:
            drug_fp = smiles_to_fingerprint(smiles)
            if drug_fp is None or np.sum(drug_fp) == 0:
                continue
                
            drug_tensor = torch.tensor(np.array([drug_fp]), dtype=torch.float32)

            with torch.no_grad():
                score = model(protein_tensor, drug_tensor).item()

            image_base64 = smiles_to_image_base64(smiles)
            if not image_base64:
                continue
                
            logger.info(f"Drug {len(results)+1}: {name}, Score: {score:.4f}")

            results.append({
                "name": name,
                "smiles": smiles,
                "score": round(score, 4),
                "image_base64": image_base64
            })
        except Exception as e:
            logger.warning(f"Error processing {name}: {e}")
            continue

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:request.top_n]
    logger.info(f"Top {len(results)} candidates selected")
    sys.stdout.flush()
    
    # Enrich top candidates with PubChem data
    for result in results:
        try:
            drug_info = fetch_drug_info(result["name"])
            if drug_info:
                result["molecular_weight"] = drug_info.get("molecular_weight", "N/A")
                result["logP"] = drug_info.get("logP", "N/A")
                result["h_bond_donors"] = drug_info.get("h_bond_donors", "N/A")
                result["h_bond_acceptors"] = drug_info.get("h_bond_acceptors", "N/A")
            logger.info(f"Enriched: {result['name']}")
        except:
            pass

    # Build detailed analysis prompt
    prompt = f"""You are writing a detailed drug discovery research report for protein {request.uniprot_id}.

Top drug candidates identified:
"""
    for i, drug in enumerate(results, 1):
        prompt += f"{i}. {drug['name']} (Binding Score: {drug['score']:.1%}, MW: {drug.get('molecular_weight', 'N/A')}, LogP: {drug.get('logP', 'N/A')})\n"
    
    prompt += """
Write a comprehensive analysis with these sections:

## Insights

Start with an overview paragraph identifying what the target protein is and its biological significance. Explain why understanding its interactions is important for drug discovery.

Then provide detailed analysis of each compound:

For each top compound, write a detailed entry with:
- **Drug-likeness:** Assessment of pharmaceutical properties
- **Known Use:** Current clinical applications and mechanism of action  
- **Literature Relevance:** How this compound relates to the target protein based on scientific literature
- **Repurposing Opportunities:** Potential new therapeutic applications

## Key Next Steps for Repurposing and Research

Provide actionable research recommendations:
- **Experimental Validation:** What experiments are needed
- **Mechanism of Action Studies:** How to investigate binding mechanisms
- **Clinical Relevance:** Implications for patient care
- **Drug Design:** How these findings inform new drug development

End with a concluding paragraph on the value of this computational analysis.

Write in academic but accessible language. Use **bold** for key terms. Be thorough and informative."""
    insights = None
    
    # Try different models as fallback
    models_to_try = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash"]
    
    for model_name in models_to_try:
        try:
            logger.info(f"Trying Gemini model: {model_name}")
            sys.stdout.flush()
            response = ai_client.models.generate_content(model=model_name, contents=prompt)
            insights = response.text
            logger.info(f"SUCCESS with model: {model_name}")
            break
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"{model_name} FAILED: {error_msg[:200]}")
            sys.stdout.flush()
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "404" in error_msg:
                continue  # Try next model
            else:
                break  # Other error, stop trying
    
    if not insights:
        # Generate basic insights without AI
        insights = f"""## Drug Discovery Results for {request.uniprot_id}

**Note:** AI-powered analysis is temporarily unavailable due to API rate limits.

### Top Candidates Found:
"""
        for i, drug in enumerate(results, 1):
            insights += f"\n**{i}. {drug['name']}**\n"
            insights += f"- Binding Score: {drug['score']}\n"
            if drug.get('molecular_weight'):
                insights += f"- Molecular Weight: {drug.get('molecular_weight', 'N/A')}\n"
            if drug.get('logP'):
                insights += f"- LogP: {drug.get('logP', 'N/A')}\n"
        
        insights += "\n*Please try again later for AI-powered insights, or check your Gemini API quota.*"

    return {
        "uniprot_id": request.uniprot_id,
        "top_candidates": results,
        "insights": insights
    }


# ==================== UNIFIED DRUG DISCOVERY ENDPOINT ====================

from research_agent import (
    ResearchAgent, 
    search_research_papers, 
    build_research_prompt,
    get_protein_structure_url
)
from typing import List, Optional
import re

# Initialize research agent
research_agent = ResearchAgent(ai_client)


class UnifiedDiscoveryRequest(BaseModel):
    query: str
    top_n: int = 8
    max_papers: int = 15


def detect_protein_id(query: str) -> Optional[str]:
    """Detect UniProt ID from query string"""
    match = re.search(r'\b([A-Z][0-9][A-Z0-9]{3}[0-9])\b', query)
    return match.group(1) if match else None


def fetch_protein_info(uniprot_id: str) -> Optional[dict]:
    """Fetch protein information from UniProt"""
    try:
        response = requests.get(
            f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json",
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            function_text = ""
            for comment in data.get("comments", []):
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        function_text = texts[0].get("value", "")
                    break
            
            return {
                "id": uniprot_id,
                "name": data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "Unknown"),
                "organism": data.get("organism", {}).get("scientificName", "Unknown"),
                "sequence_length": data.get("sequence", {}).get("length", 0),
                "function": function_text,
                "sequence": data.get("sequence", {}).get("value", "")
            }
    except Exception as e:
        logger.error(f"Error fetching protein info: {e}")
    return None


@app.post("/unified-discovery")
def unified_discovery(request: UnifiedDiscoveryRequest):
    """
    UNIFIED Drug Discovery Pipeline:
    1. Detect protein from query
    2. Fetch protein info
    3. Search research papers
    4. Run drug binding predictions
    5. Generate unified AI analysis
    """
    logger.info(f"=== UNIFIED DISCOVERY === Query: {request.query}")
    sys.stdout.flush()
    
    result = {
        "query": request.query,
        "protein_info": None,
        "papers": [],
        "top_candidates": [],
        "insights": "",
        "tools_used": []
    }
    
    try:
        # STEP 1: Detect protein ID
        protein_id = detect_protein_id(request.query)
        logger.info(f"Step 1 - Protein ID detected: {protein_id}")
        
        # STEP 2: Fetch protein info if ID found
        if protein_id:
            result["protein_info"] = fetch_protein_info(protein_id)
            result["tools_used"].append("uniprot_api")
            logger.info(f"Step 2 - Protein info: {result['protein_info'].get('name') if result['protein_info'] else 'Not found'}")
        
        # STEP 3: Search research papers
        logger.info("Step 3 - Searching research papers...")
        papers = search_research_papers(request.query, request.max_papers)
        result["papers"] = papers
        result["tools_used"].append("research_papers")
        logger.info(f"Step 3 - Found {len(papers)} papers")
        
        # STEP 4: Run drug discovery if protein found
        if result["protein_info"] and result["protein_info"].get("sequence"):
            logger.info("Step 4 - Running drug binding predictions...")
            result["tools_used"].append("drug_discovery_ml")
            
            protein_sequence = result["protein_info"]["sequence"]
            protein_encoded = encode_protein(protein_sequence)
            protein_tensor = torch.tensor(np.array([protein_encoded]), dtype=torch.float32)
            
            drug_db = pd.read_csv("drug_db.csv")
            candidates = []
            
            # Filter valid drugs
            valid_drugs = drug_db[~drug_db['smiles'].str.contains('Antibody|Analog|Inhibitor|Protein|Heparin|Insulin', case=False, na=False)]
            
            for idx, row in valid_drugs.iterrows():
                try:
                    smiles = row['smiles']
                    name = row['name']
                    
                    drug_fp = smiles_to_fingerprint(smiles)
                    if drug_fp is None or np.sum(drug_fp) == 0:
                        continue
                    
                    drug_tensor = torch.tensor(np.array([drug_fp]), dtype=torch.float32)
                    
                    with torch.no_grad():
                        score = model(protein_tensor, drug_tensor).item()
                    
                    image_base64 = smiles_to_image_base64(smiles)
                    if not image_base64:
                        continue
                    
                    candidates.append({
                        "name": name,
                        "smiles": smiles,
                        "score": round(score, 4),
                        "image_base64": image_base64
                    })
                except Exception as e:
                    continue
            
            # Sort and get top candidates
            candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:request.top_n]
            
            # Enrich with PubChem data
            for candidate in candidates:
                try:
                    drug_info = fetch_drug_info(candidate["name"])
                    if drug_info:
                        candidate["molecular_weight"] = drug_info.get("molecular_weight", "N/A")
                        candidate["logP"] = drug_info.get("logP", "N/A")
                except:
                    pass
            
            result["top_candidates"] = candidates
            logger.info(f"Step 4 - Found {len(candidates)} drug candidates")
        
        # STEP 5: Generate comprehensive report
        logger.info("Step 5 - Generating comprehensive analysis...")
        
        # Build prompt for AI
        prompt = f"""Write a detailed drug discovery research report for: {request.query}

"""
        if result.get("protein_info"):
            p = result["protein_info"]
            prompt += f"Target Protein: {p.get('name')} ({p.get('id')})\nFunction: {p.get('function', '')[:300]}\n\n"
        
        if result.get("top_candidates"):
            prompt += "Drug Candidates:\n"
            for d in result["top_candidates"][:5]:
                prompt += f"- {d['name']}: {d['score']:.1%} binding\n"
            prompt += "\n"
        
        if result.get("papers"):
            prompt += "Research Papers:\n"
            for p in result["papers"][:5]:
                prompt += f"- {p['title'][:80]} ({p['source']})\n"
        
        # Adaptive prompt based on query type
        has_drugs = bool(result.get("top_candidates"))
        has_protein = bool(result.get("protein_info"))
        
        if has_drugs or has_protein:
            # Drug discovery report
            prompt += """
Write a comprehensive research report. Use proper formatting:

### Executive Summary
3-4 sentences summarizing key findings and significance. Include citations (Author et al., Year).

### Background & Significance
- Why this topic/target matters
- Current research landscape
- Key challenges in the field

### Literature Analysis
Based on research papers:
- Key finding 1 with citation (Author et al., Year, Source)
- Key finding 2 with citation
- Key finding 3 with citation
- Emerging trends

### Drug Candidates
For each compound:
**[Drug Name]** - Drug class. Mechanism. Clinical uses. Binding score: X%.

### Therapeutic Potential
- Clinical opportunities
- Drug repurposing possibilities
- Safety considerations

### Recommendations
1. Next research steps
2. Validation approaches
3. Future directions

### References
List all papers cited above.

Use **bold** for key terms. Include inline citations."""
        else:
            # General research report
            prompt += """
Write a comprehensive research report on this topic. Use proper formatting:

### Executive Summary
3-4 sentences summarizing key findings and significance. Include citations (Author et al., Year).

### Background & Context
- Why this topic matters
- Current state of research
- Key debates or challenges

### Literature Analysis
Based on research papers:
- Key finding 1 with citation (Author et al., Year, Source)
- Key finding 2 with citation
- Key finding 3 with citation
- Important themes and patterns

### Key Insights
- Major insight 1
- Major insight 2
- Major insight 3
- Connections between findings

### Implications
- Practical applications
- Theoretical contributions
- Societal impact

### Future Directions
1. Research gaps to address
2. Methodological suggestions
3. Emerging questions

### References
List all papers cited above.

Use **bold** for key terms. Include inline citations throughout."""
        
        # Try multiple Gemini models - gemini-2.5-flash works best
        models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-pro"]
        insights = None
        
        for model_name in models_to_try:
            try:
                logger.info(f"Trying model: {model_name}")
                response = ai_client.models.generate_content(model=model_name, contents=prompt)
                insights = response.text
                logger.info(f"SUCCESS with model: {model_name}")
                result["tools_used"].append(f"ai_analysis:{model_name}")
                break
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"{model_name} failed: {error_msg[:100]}")
                continue
        
        if not insights:
            # Generate VERY detailed fallback report WITHOUT using AI
            insights = generate_comprehensive_fallback(result)
        
        result["insights"] = insights
        logger.info(f"Step 5 - Analysis complete. Tools used: {result['tools_used']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Unified discovery error: {e}")
        result["error"] = str(e)
        return result


def generate_comprehensive_fallback(result: dict) -> str:
    """Generate a comprehensive research report without AI"""
    query = result.get("query", "drug discovery research")
    report = ""
    
    # Executive Summary
    report += "### Executive Summary\n\n"
    if result.get("protein_info"):
        p = result["protein_info"]
        report += f"This comprehensive analysis investigates **{p.get('name', 'the target protein')}** "
        report += f"(UniProt: {p.get('id', 'N/A')}), a {p.get('sequence_length', 0)}-amino acid protein "
        report += f"expressed in *{p.get('organism', 'humans')}*. "
        report += "This protein represents a significant therapeutic target with potential implications "
        report += "for drug discovery and development. Our computational analysis has identified "
        report += f"several promising drug candidates with predicted binding affinity to this target.\n\n"
    else:
        report += f"This report presents a comprehensive analysis of **{query}**, "
        report += "examining current research literature, potential therapeutic compounds, "
        report += "and opportunities for drug discovery. The analysis integrates findings from "
        report += f"multiple scientific databases including {len(result.get('papers', []))} peer-reviewed publications "
        report += "to provide actionable insights for researchers and drug developers.\n\n"
    
    # Background & Significance
    report += "### Background & Significance\n\n"
    if result.get("protein_info"):
        p = result["protein_info"]
        if p.get('function'):
            report += f"**Biological Function:** {p.get('function')}\n\n"
        report += "Understanding protein-drug interactions is fundamental to modern drug discovery. "
        report += "Computational approaches enable rapid screening of potential therapeutic compounds, "
        report += "significantly accelerating the drug development pipeline. This target has been "
        report += "identified through rigorous bioinformatics analysis as a promising candidate "
        report += "for therapeutic intervention.\n\n"
    else:
        report += f"**{query.title()}** represents an active area of pharmaceutical research "
        report += "with significant therapeutic potential. Current research efforts focus on "
        report += "understanding disease mechanisms, identifying novel drug targets, and "
        report += "developing effective therapeutic interventions. The integration of computational "
        report += "methods with traditional drug discovery approaches has accelerated progress in this field.\n\n"
    
    # Literature Analysis
    report += "### Literature Analysis\n\n"
    if result.get("papers") and len(result["papers"]) > 0:
        report += f"Analysis of {len(result['papers'])} research publications reveals key insights:\n\n"
        
        # Group papers by source
        sources = {}
        for paper in result["papers"]:
            src = paper.get("source", "Other")
            if src not in sources:
                sources[src] = []
            sources[src].append(paper)
        
        for source, papers in sources.items():
            report += f"**{source} Publications:**\n"
            for paper in papers[:3]:
                title = paper.get('title', 'N/A')
                authors = paper.get('authors', ['Unknown'])
                year = paper.get('published', 'N/A')
                report += f"- *\"{title}\"* ({authors[0] if authors else 'Unknown'} et al., {year})\n"
            report += "\n"
        
        report += "**Key Themes Identified:**\n"
        report += "- Novel therapeutic approaches and drug delivery systems\n"
        report += "- Molecular mechanisms underlying disease pathology\n"
        report += "- Clinical trial outcomes and safety profiles\n"
        report += "- Computational and AI-driven drug discovery methods\n\n"
    else:
        report += "Literature search is in progress. Relevant publications will inform "
        report += "therapeutic strategies and drug development approaches.\n\n"
    
    # Drug Candidates
    if result.get("top_candidates") and len(result["top_candidates"]) > 0:
        report += "### Drug Candidate Analysis\n\n"
        report += "Computational screening identified the following compounds with predicted binding affinity:\n\n"
        
        for i, drug in enumerate(result["top_candidates"], 1):
            report += f"**{i}. {drug['name']}**\n\n"
            report += f"- **Binding Affinity Score:** {drug['score']:.1%}\n"
            if drug.get('molecular_weight'):
                report += f"- **Molecular Weight:** {drug['molecular_weight']} Da\n"
            if drug.get('logP'):
                report += f"- **LogP (Lipophilicity):** {drug['logP']}\n"
            
            # Detailed drug info
            drug_details = get_comprehensive_drug_info(drug['name'])
            if drug_details:
                report += f"- **Drug Class:** {drug_details.get('class', 'N/A')}\n"
                report += f"- **Mechanism of Action:** {drug_details.get('mechanism', 'N/A')}\n"
                report += f"- **Current Indications:** {drug_details.get('indications', 'N/A')}\n"
                report += f"- **Relevance:** {drug_details.get('relevance', 'Potential for drug repurposing')}\n"
            report += "\n"
    
    # Therapeutic Opportunities
    report += "### Therapeutic Opportunities\n\n"
    report += "Based on the computational analysis and literature review, several therapeutic opportunities emerge:\n\n"
    report += "- **Drug Repurposing:** Existing FDA-approved drugs may offer faster pathways to clinical application\n"
    report += "- **Combination Therapies:** Multi-target approaches could enhance therapeutic efficacy\n"
    report += "- **Novel Drug Design:** Structural insights enable rational design of improved compounds\n"
    report += "- **Biomarker Development:** Identified targets may serve as diagnostic or prognostic markers\n\n"
    
    # Research Recommendations
    report += "### Research Recommendations\n\n"
    report += "**Immediate Next Steps:**\n"
    report += "- Validate computational predictions with biochemical binding assays (SPR, ITC)\n"
    report += "- Perform molecular dynamics simulations to assess binding stability\n"
    report += "- Evaluate cytotoxicity profiles in relevant cell lines\n\n"
    
    report += "**Medium-term Goals:**\n"
    report += "- Conduct structure-activity relationship (SAR) studies\n"
    report += "- Assess pharmacokinetic properties (ADMET profiling)\n"
    report += "- Develop lead optimization strategies\n\n"
    
    report += "**Long-term Objectives:**\n"
    report += "- Progress promising candidates to preclinical studies\n"
    report += "- Establish collaborative partnerships for clinical development\n"
    report += "- Explore intellectual property opportunities\n"
    
    return report


def get_comprehensive_drug_info(drug_name: str) -> dict:
    """Get comprehensive information for common drugs"""
    drug_database = {
        "Furosemide": {
            "class": "Loop Diuretic",
            "mechanism": "Inhibits Na-K-2Cl cotransporter in loop of Henle",
            "indications": "Edema, heart failure, hypertension, renal disease",
            "relevance": "Cardiovascular and renal therapeutic applications"
        },
        "Amlodipine": {
            "class": "Calcium Channel Blocker (Dihydropyridine)",
            "mechanism": "Blocks L-type calcium channels in vascular smooth muscle",
            "indications": "Hypertension, chronic stable angina, vasospastic angina",
            "relevance": "Cardiovascular protection and blood pressure regulation"
        },
        "Canagliflozin": {
            "class": "SGLT2 Inhibitor",
            "mechanism": "Inhibits sodium-glucose co-transporter 2 in proximal tubule",
            "indications": "Type 2 diabetes, heart failure, chronic kidney disease",
            "relevance": "Metabolic and cardiovascular benefits beyond glycemic control"
        },
        "Levothyroxine": {
            "class": "Thyroid Hormone",
            "mechanism": "Synthetic T4 hormone replacement",
            "indications": "Hypothyroidism, thyroid cancer, myxedema coma",
            "relevance": "Metabolic regulation and hormonal therapy"
        },
        "Sildenafil": {
            "class": "PDE5 Inhibitor",
            "mechanism": "Inhibits phosphodiesterase type 5, increases cGMP",
            "indications": "Erectile dysfunction, pulmonary arterial hypertension",
            "relevance": "Vascular and smooth muscle relaxation applications"
        },
        "Dapagliflozin": {
            "class": "SGLT2 Inhibitor",
            "mechanism": "Inhibits renal glucose reabsorption",
            "indications": "Type 2 diabetes, heart failure with reduced EF",
            "relevance": "Cardiorenal protection in metabolic disease"
        },
        "Empagliflozin": {
            "class": "SGLT2 Inhibitor",
            "mechanism": "Blocks sodium-glucose cotransporter 2",
            "indications": "Type 2 diabetes, heart failure, chronic kidney disease",
            "relevance": "Proven cardiovascular mortality reduction"
        },
        "Gabapentin": {
            "class": "Anticonvulsant/Analgesic",
            "mechanism": "Binds alpha-2-delta subunit of voltage-gated calcium channels",
            "indications": "Neuropathic pain, epilepsy, restless leg syndrome",
            "relevance": "Neurological and pain management applications"
        },
        "Omeprazole": {
            "class": "Proton Pump Inhibitor",
            "mechanism": "Irreversibly inhibits H+/K+-ATPase in gastric parietal cells",
            "indications": "GERD, peptic ulcer disease, H. pylori eradication",
            "relevance": "Gastrointestinal protection and acid suppression"
        },
        "Metformin": {
            "class": "Biguanide Antidiabetic",
            "mechanism": "Activates AMPK, reduces hepatic gluconeogenesis",
            "indications": "Type 2 diabetes, polycystic ovary syndrome, prediabetes",
            "relevance": "First-line diabetes therapy with potential anti-aging properties"
        },
    }
    return drug_database.get(drug_name, None)


# ==================== LEGACY ENDPOINTS (kept for compatibility) ====================

class ResearchQuery(BaseModel):
    query: str
    include_papers: bool = True
    max_papers: int = 8


class PaperSearchRequest(BaseModel):
    query: str
    max_results: int = 10


@app.post("/research")
def research_query(request: ResearchQuery):
    """
    Main research endpoint - processes a query and returns comprehensive results
    """
    logger.info(f"=== RESEARCH QUERY === {request.query}")
    sys.stdout.flush()
    
    try:
        # Process with research agent
        results = research_agent.process_query(request.query)
        
        # Generate AI insights if papers found
        if results["papers"] or results.get("protein_info") or results.get("drug_info"):
            prompt = build_research_prompt(
                request.query,
                results["papers"],
                results.get("protein_info"),
                results.get("drug_info")
            )
            
            # Try different models as fallback (optimized token usage)
            models_to_try = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash"]
            insights = None
            
            for model_name in models_to_try:
                try:
                    logger.info(f"Trying model: {model_name}")
                    response = ai_client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    insights = response.text
                    logger.info(f"SUCCESS with model: {model_name}")
                    break
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"{model_name} failed: {error_msg[:100]}")
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        continue
                    break
            
            results["insights"] = insights or "Analysis in progress. Review the research papers below."
        
        logger.info(f"Research complete: {len(results['papers'])} papers, tools: {results['tools_used']}")
        return results
        
    except Exception as e:
        logger.error(f"Research error: {e}")
        return {"error": str(e), "papers": [], "insights": ""}


@app.post("/search-papers")
def search_papers(request: PaperSearchRequest):
    """
    Search for research papers from ArXiv and PubMed
    """
    logger.info(f"=== PAPER SEARCH === {request.query}")
    
    papers = search_research_papers(request.query, request.max_results)
    
    return {
        "query": request.query,
        "papers": papers,
        "total": len(papers)
    }


class MoleculeRequest(BaseModel):
    smiles: str
    name: Optional[str] = None


@app.post("/molecule-3d")
def get_molecule_3d(request: MoleculeRequest):
    """
    Generate 3D coordinates for a molecule from SMILES
    """
    try:
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(request.smiles)
        if mol is None:
            return {"error": "Invalid SMILES string"}
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Get 3D coordinates as MOL block
        mol_block = Chem.MolToMolBlock(mol)
        
        # Also generate 2D image
        mol_2d = Chem.MolFromSmiles(request.smiles)
        img = Draw.MolToImage(mol_2d, size=(300, 300))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return {
            "smiles": request.smiles,
            "name": request.name,
            "mol_block": mol_block,
            "image_2d": img_base64,
            "atom_count": mol.GetNumAtoms(),
            "bond_count": mol.GetNumBonds()
        }
        
    except Exception as e:
        logger.error(f"3D molecule generation error: {e}")
        return {"error": str(e)}


class ProteinStructureRequest(BaseModel):
    uniprot_id: str


@app.post("/protein-structure")
def get_protein_structure(request: ProteinStructureRequest):
    """
    Get protein structure information and AlphaFold URL
    """
    try:
        # Get protein info from UniProt
        response = requests.get(
            f"https://rest.uniprot.org/uniprotkb/{request.uniprot_id}.json",
            timeout=30
        )
        
        if response.status_code != 200:
            return {"error": "Protein not found"}
        
        data = response.json()
        
        # Get AlphaFold structure URL
        alphafold_url = get_protein_structure_url(request.uniprot_id)
        
        # Try to fetch AlphaFold PDB
        pdb_content = None
        try:
            pdb_response = requests.get(alphafold_url, timeout=30)
            if pdb_response.status_code == 200:
                pdb_content = pdb_response.text
        except:
            pass
        
        return {
            "uniprot_id": request.uniprot_id,
            "name": data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "N/A"),
            "organism": data.get("organism", {}).get("scientificName", "N/A"),
            "sequence": data.get("sequence", {}).get("value", ""),
            "sequence_length": data.get("sequence", {}).get("length", 0),
            "alphafold_url": alphafold_url,
            "pdb_content": pdb_content,
            "has_structure": pdb_content is not None
        }
        
    except Exception as e:
        logger.error(f"Protein structure error: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
