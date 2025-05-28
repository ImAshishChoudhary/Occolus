# OccolusAI

OccolusAI is an intelligent, AI-powered platform designed to accelerate protein-based drug discovery. With cutting-edge ML models, real-time data integrations, and LLM-driven insights, OccolusAI empowers researchers, educators, and biotech professionals to explore protein-drug interactions like never before.

🌐 **[Live Demo](https://moledrugs.vercel.app/)**

---

## 🚀 Overview

OccolusAI is a lightweight protein-focused drug discovery tool that uses AI/ML models and large language models (LLMs) to support:

- Protein structure analysis  
- Binding site prediction  
- Ligand screening  
- Automated literature and data insights  

Our goal is to lower barriers to entry in early-stage drug discovery, making these tools accessible to smaller labs, academic researchers, and biotech startups.

---

## 🔬 Key Features

✅ **Drug-Target Interaction (DTI) Prediction**  
Predict the probability of a drug binding to a protein target using a robust ML pipeline.

✅ **Protein & Drug Feature Extraction**  
- **Protein Encoding:** Uses ProtBERT embeddings for structural and functional context.  
- **Drug Encoding:** Uses RDKit Morgan fingerprints (ECFP4) to capture molecular features.  
- **Prediction:** A fully connected feedforward neural network concatenates these embeddings to predict binding probabilities.

✅ **LLM-Driven Natural Language Insights**  
Analyze and interpret findings using Gemini AI to generate natural-language summaries and explanations.

✅ **Real-Time Data Integration**  
Automatically fetch and curate data from trusted sources:
- **PubChem** for drug compounds  
- **UniProt** for protein sequences

✅ **Interactive Visualizations**  
- Molecular structure viewers  
- Interaction heatmaps  
- Intuitive dashboards to streamline exploration

✅ **Plug & Play Extensibility**  
- Integrate your own models, datasets, or Jupyter notebooks  
- Experiment within sandboxed environments for safe testing

✅ **Discovery of Novel Candidates**  
Find potential new drugs for any protein target with a powerful search and analysis pipeline.

---

## 👥 Target Users

- **Researchers:** Scientists in pharmacology, biotech, and molecular biology.  
- **Students & Educators:** Use OccolusAI for advanced educational purposes and demonstrations.  
- **Biotech Startups:** Leverage AI-driven tools without heavy infrastructure costs.  
- **Pharmaceutical Professionals:** Support early-stage drug development.  
- **Policy Makers & Health Organizations:** Explore applications for healthcare planning and global health initiatives.

---

## 🌎 Impact & Real-World Relevance

- **Accelerate early-stage drug target exploration.**  
- **Enable therapeutic discovery for diseases like cancer and neurodegeneration.**  
- **Reduce barriers for smaller teams and academic labs.**  
- **Foster collaboration and open science initiatives.**

---

## 💡 Monetization & Future Vision

- **Freemium Model:**  
  - Free tier for researchers, students, and open science projects.  
  - Pro tier for startups and biotech firms with advanced features like private project hosting and custom model integration.

- **Consulting & Integration Services:**  
  - Tailored deployments and training for research labs and biotech companies.

- **Community Plugins/Marketplace:**  
  - Allow users to create, share, and sell their own models and plugins within the OccolusAI ecosystem.

---

## ⚙️ Tech Stack

- **Frontend:** Next.js  
- **Backend:** FastAPI (Python)  
- **ML Models:** PyTorch, RDKit, ProtBERT  
- **LLM Integration:** Gemini AI  

---

## ⚙️ Setup Guide

### Environment Variables Reference

**Server `.env` (reference only):**
```bash
# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000

# Model Configuration
MODEL_PATH=drug_target_model.pth
```

**Client `.env` (reference only):**
```bash
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000

# Environment
NODE_ENV=development
```

---

### Local Development

#### Server
```bash
cd server
pip install -r requirements.txt
uvicorn main:app --reload
```

#### Client
```bash
cd client
npm install
npm run dev
```

---

## 🔭 Scalability & Future Scope

- Integrate more protein databases and real-time structure predictions  
- Incorporate advanced molecular docking and simulation tools  
- Expand LLM features for automated report generation  
- Add multi-target drug screening capabilities  
- Build collaborative workspaces for sharing results and workflows  

---

## 🤝 Contributing

We welcome contributions! If you’re interested in improving OccolusAI or adding new features, feel free to open an issue or submit a pull request.

---

## 📜 License

This project is open-source under the MIT License.

---

## 🙏 Acknowledgments

### 👥 Team Members
- Gauri
- Ashish K Choudhary
- Mohit Taneja 

---