# Occolus

An open-source drug discovery platform that leverages machine learning and AI to accelerate the identification of therapeutic compounds. Occolus bridges the gap between protein targets and potential drug candidates by providing researchers with intelligent screening, molecular analysis, and automated research synthesis.

---

## The Problem

Drug discovery remains one of the most expensive and time-consuming endeavors in science. Traditional approaches require years of manual screening, with success rates below 10%. Researchers face challenges in:

- Identifying promising drug candidates from vast chemical libraries
- Predicting how compounds will interact with target proteins
- Assessing drug-likeness and safety profiles early in development
- Synthesizing insights from thousands of research papers
- Visualizing and communicating molecular structures effectively

Occolus addresses these challenges by combining computational chemistry with modern AI to provide a unified discovery workflow.

---

## What Occolus Does

### Intelligent Target Discovery

Search for any protein using UniProt IDs, gene names, or disease associations. Occolus retrieves comprehensive protein data including amino acid sequences, functional annotations, organism information, and known binding sites. The platform automatically identifies relevant drug targets and maps them to existing therapeutic compounds.

### Compound Screening and Ranking

Screen FDA-approved drugs and experimental compounds against your target protein. The machine learning model predicts binding affinity using molecular fingerprints and protein sequence encoding, ranking candidates by their likelihood of therapeutic interaction. Each prediction includes confidence scores and molecular property analysis.

### ADMET Property Prediction

Evaluate drug-likeness before investing in expensive lab work. Occolus calculates absorption, distribution, metabolism, excretion, and toxicity indicators including Lipinski's Rule of Five compliance.

### Molecular Visualization

View compounds in both 2D structural diagrams and interactive 3D representations. Explore bond angles, functional groups, and spatial configurations.

### Analog Generation

Generate structural analogs of promising compounds. The platform applies chemical transformations to explore nearby chemical space, identifying derivatives that may offer improved binding or reduced toxicity.

### AI-Powered Research Reports

Synthesize findings into comprehensive research reports. Occolus queries scientific literature, extracts relevant insights, and generates structured analyses with proper citations. Export as PDF or Word documents.

---

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│     Frontend     │────▶│     Backend      │────▶│   External APIs  │
│    (Next.js)     │◀────│    (FastAPI)     │◀────│                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            ┌──────────────┐        ┌──────────────┐
            │   PyTorch    │        │    RDKit     │
            │   ML Model   │        │  Chemistry   │
            └──────────────┘        └──────────────┘
```

**Data Flow:**
1. User submits protein/disease query
2. Backend fetches protein data from UniProt
3. ML model predicts binding affinity for compounds
4. RDKit calculates molecular properties and generates structures
5. Gemini AI synthesizes research insights
6. Results rendered with 2D/3D visualization

---

## ML Model

The binding prediction model uses a feedforward neural network:

```python
class DrugTargetModel(nn.Module):
    def __init__(self, protein_dim=400, drug_dim=1024):
        super().__init__()
        self.fc1 = nn.Linear(protein_dim + drug_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))
```

**Input Features:**
- Protein sequence encoding: 400 dimensions (20 amino acids × 20 positions)
- Morgan fingerprints: 1024-bit molecular representation via RDKit

**Output:**
- Binding probability score (0-1)

---

## Project Structure

```
occolus/
├── client/
│   ├── app/
│   │   ├── page.tsx              # Main UI component
│   │   ├── layout.tsx            # Root layout
│   │   └── globals.css
│   └── components/
│       └── Molecule3DViewer.tsx  # 3Dmol.js wrapper
│
├── server/
│   ├── main.py                   # FastAPI endpoints
│   ├── drug_discovery.py         # ADMET, analogs, docking logic
│   ├── drug_db.csv               # 150+ FDA-approved compounds
│   ├── drug_target_model.pth     # Trained PyTorch model
│   └── requirements.txt
│
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.11+, Node.js 18+, Google Gemini API key

```bash
# Clone
git clone https://github.com/your-username/occolus.git
cd occolus

# Backend
cd server
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# Create .env
echo "GEMINI_API_KEY=your_key_here" > .env

# Frontend
cd ../client
npm install
```

**Run:**

```bash
# Terminal 1 - Backend
cd server
uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend
cd client
npm run dev
```

Open http://localhost:3000

---

## API Endpoints

### Unified Discovery

```bash
curl -X POST http://localhost:8000/unified-discovery \
  -H "Content-Type: application/json" \
  -d '{"query": "EGFR inhibitors"}'
```

Response includes `protein_info`, `top_candidates`, `insights`, and `papers`.

### ADMET Analysis

```bash
curl -X POST http://localhost:8000/admet \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}'
```

Returns molecular weight, LogP, HBD/HBA, TPSA, and Lipinski compliance.

### Generate Analogs

```bash
curl -X POST http://localhost:8000/generate-analogs \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "num_analogs": 5}'
```

Returns SMILES and properties for structural derivatives.

### 3D Coordinates

```bash
curl -X POST http://localhost:8000/molecule-3d-coords \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}'
```

Returns MOL block for 3Dmol.js rendering.

### Export Reports

```bash
curl -X POST http://localhost:8000/export-pdf \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "insights": "...", "top_candidates": [...]}' \
  --output report.pdf
```

---

## Core Dependencies

**Backend:**
```
fastapi==0.115.0
torch==2.6.0
rdkit==2024.9.5
google-generativeai==1.9.0
pandas==2.2.3
reportlab==4.0.8
python-docx==1.1.0
```

**Frontend:**
```
next@14.0.0
react@18.2.0
typescript@5.2.0
tailwindcss@3.3.0
framer-motion@11.0.0
3dmol@2.0.0
```

---

## Roadmap

**Current**
- Protein target search via UniProt
- Drug screening with ML binding prediction
- ADMET property calculation
- 2D/3D molecular visualization
- AI research synthesis with Gemini
- PDF/DOCX report export

**Planned**
- Molecular docking simulation
- De novo drug generation
- Multi-target screening
- Batch processing API

---

## Contributing

```bash
# Fork and clone
git clone https://github.com/your-username/occolus.git

# Create branch
git checkout -b feature/your-feature

# Make changes, then commit
git commit -m "Add feature"

# Push and open PR
git push origin feature/your-feature
```

Areas for contribution:
- Expand compound database (ChEMBL integration)
- Improve ML model (graph neural networks)
- Add molecular docking (AutoDock Vina)
- Enhance 3D visualization

---

Built for researchers. Contributions welcome.
