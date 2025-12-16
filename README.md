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

Evaluate drug-likeness before investing in expensive lab work. Occolus calculates:

- **Absorption** — Lipophilicity (LogP), topological polar surface area
- **Distribution** — Molecular weight, hydrogen bond donors and acceptors
- **Metabolism** — Rotatable bonds, aromatic ring count
- **Excretion** — Molecular complexity indicators
- **Toxicity** — Lipinski's Rule of Five compliance, structural alerts

### Molecular Visualization

View compounds in both 2D structural diagrams and interactive 3D representations. Explore bond angles, functional groups, and spatial configurations. Compare structural similarities between lead compounds and their analogs.

### Analog Generation

Generate structural analogs of promising compounds. The platform applies chemical transformations to explore nearby chemical space, identifying derivatives that may offer improved binding or reduced toxicity.

### AI-Powered Research Reports

Synthesize findings into comprehensive research reports. Occolus queries scientific literature, extracts relevant insights, and generates structured analyses covering:

- Target protein biology and therapeutic relevance
- Mechanism of action for identified compounds
- Existing clinical evidence and trial data
- Potential drug repurposing opportunities
- Recommended next steps for lead optimization

Export reports as professionally formatted PDF or Word documents with proper citations.

---

## Features

| Feature | Description |
|---------|-------------|
| Protein Search | Query by UniProt ID, gene name, or disease keyword |
| Drug Screening | Rank compounds by predicted binding affinity |
| ADMET Analysis | Evaluate drug-likeness and safety indicators |
| 2D/3D Visualization | Interactive molecular structure viewing |
| Analog Generation | Explore structural derivatives of lead compounds |
| Literature Synthesis | AI-generated insights from research papers |
| Report Export | Download findings as PDF or DOCX |
| Compound Database | 150+ FDA-approved drugs with full molecular data |

---

## Screenshots

The landing page provides a clean search interface for protein and disease queries:

```
[Landing Page - Search proteins, diseases, drug targets]
```

Research results display identified compounds ranked by binding score alongside AI-generated insights:

```
[Research View - Left sidebar with stats, center with report, right with references]
```

Compound details show molecular structure, properties, and ADMET predictions:

```
[Drug Detail Modal - 2D/3D structure toggle, properties table, analog list]
```

---

## Technology Stack

### Backend

| Component | Technology | Purpose |
|-----------|------------|---------|
| API Framework | FastAPI | High-performance async REST API |
| ML Framework | PyTorch | Neural network inference |
| Cheminformatics | RDKit | Molecular fingerprints, property calculation, structure generation |
| AI Analysis | Google Gemini | Research synthesis and insight generation |
| Data Processing | Pandas, NumPy | Molecular data manipulation |

### Frontend

| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | Next.js 13+ | React with App Router |
| Language | TypeScript | Type-safe development |
| Styling | Tailwind CSS | Utility-first CSS |
| Animation | Framer Motion | Smooth UI transitions |
| 3D Rendering | 3Dmol.js | Molecular visualization |

### External APIs

| Service | Purpose |
|---------|---------|
| UniProt | Protein sequences and annotations |
| PubChem | Chemical compound data |
| Europe PMC | Scientific literature |
| AlphaFold DB | Protein 3D structures |

---

## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- Git
- Google Gemini API key (free tier available)

### Clone Repository

```bash
git clone https://github.com/your-username/occolus.git
cd occolus
```

### Backend Setup

```bash
cd server

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

### Frontend Setup

```bash
cd client
npm install
```

### Run Application

Terminal 1 (Backend):
```bash
cd server
uvicorn main:app --reload --port 8000
```

Terminal 2 (Frontend):
```bash
cd client
npm run dev
```

Access the application at http://localhost:3000

API documentation available at http://localhost:8000/docs

---

## Project Structure

```
occolus/
├── client/                     # Frontend application
│   ├── app/
│   │   ├── page.tsx           # Main application page
│   │   ├── layout.tsx         # Root layout
│   │   └── globals.css        # Global styles
│   ├── components/
│   │   └── Molecule3DViewer.tsx
│   └── public/                # Static assets
│
├── server/                     # Backend application
│   ├── main.py                # API endpoints
│   ├── drug_discovery.py      # Core discovery logic
│   ├── drug_db.csv            # Compound database
│   ├── drug_target_model.pth  # Trained ML model
│   └── requirements.txt       # Python dependencies
│
└── README.md
```

---

## API Reference

### Core Endpoints

**Unified Discovery**
```http
POST /unified-discovery
Content-Type: application/json

{
  "query": "EGFR inhibitors"
}
```

Returns protein information, ranked drug candidates, research insights, and relevant papers.

**ADMET Prediction**
```http
POST /admet
Content-Type: application/json

{
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"
}
```

Returns drug-likeness assessment and property calculations.

**Generate Analogs**
```http
POST /generate-analogs
Content-Type: application/json

{
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "num_analogs": 5
}
```

Returns structurally similar compounds.

**3D Coordinates**
```http
POST /molecule-3d-coords
Content-Type: application/json

{
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"
}
```

Returns MOL block for 3D visualization.

**Export Reports**
```http
POST /export-pdf
POST /export-docx
```

Returns downloadable document with full research report.

---

## Machine Learning Model

The binding affinity prediction model uses a feedforward neural network architecture:

**Input Features**
- Protein sequence encoding (400 dimensions)
- Morgan molecular fingerprints (1024 bits)

**Architecture**
```
Input (1424) → Dense (128) → ReLU → Dense (64) → ReLU → Dense (1) → Sigmoid
```

**Output**
- Binding probability score (0-1)

The model was trained on known drug-protein interactions and provides relative rankings for compound screening. Predictions should be validated experimentally.

---

## Contributing

We welcome contributions from the community. Here's how to get involved:

### Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Test thoroughly
5. Commit with clear messages: `git commit -m "Add feature description"`
6. Push to your fork: `git push origin feature/your-feature`
7. Open a pull request

### Contribution Areas

**Expand Compound Database**
- Add more FDA-approved drugs
- Include experimental compounds from ChEMBL
- Integrate natural product libraries

**Improve ML Models**
- Implement graph neural networks for molecular representation
- Add uncertainty quantification to predictions
- Train on larger drug-target interaction datasets

**Enhance Visualization**
- Add protein-ligand docking visualization
- Implement pharmacophore mapping
- Build comparison views for multiple compounds

**Extend Analysis**
- Add synthesis route prediction
- Implement toxicity prediction models
- Build drug-drug interaction checking

### Code Standards

- Python: Follow PEP 8, use type hints
- TypeScript: Follow ESLint configuration
- Commits: Use conventional commit format
- Documentation: Update README for new features

---

## Roadmap

**Current Version**
- Protein target search and discovery
- FDA-approved drug screening
- ADMET property prediction
- 2D/3D molecular visualization
- AI research report generation
- PDF/DOCX export

**Planned Features**
- Molecular docking simulation
- De novo drug generation
- Multi-target screening
- Collaborative workspaces
- API access for programmatic use

---

## License

This project is released under the MIT License. See the LICENSE file for details.

You are free to use, modify, and distribute this software for any purpose, including commercial applications.

---

## Acknowledgments

Occolus is built on the work of many open-source projects and public databases:

- RDKit for cheminformatics
- PyTorch for machine learning
- UniProt Consortium for protein data
- PubChem for chemical information
- Europe PMC for literature access

---

## Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Join conversations in GitHub Discussions
- **Documentation**: API docs at `/docs` when server is running

---

Built for researchers, by researchers. Contributions welcome.
