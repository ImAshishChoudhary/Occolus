
# OccolusAI

OccolusAI is an intelligent, AI-powered platform designed to accelerate protein-based drug discovery. Leveraging cutting-edge ML models, real-time data integrations, and LLM-driven insights, OccolusAI empowers researchers, educators, and biotech professionals to explore protein-drug interactions like never before.

🚀 **Overview**  
OccolusAI is a lightweight, protein-focused drug discovery tool that combines AI/ML models and large language models (LLMs) to support:
- Protein structure analysis  
- Binding site prediction  
- Ligand screening  
- Automated literature and data insights  

Our mission is to lower barriers to entry in early-stage drug discovery, making advanced tools accessible to smaller labs, academic researchers, and biotech startups.

🔬 **Key Features**  
✅ **Drug-Target Interaction (DTI) Prediction**  
✅ **Protein & Drug Feature Extraction**  
✅ **LLM-Driven Natural Language Insights**  
✅ **Real-Time Data Integration**  
✅ **Interactive Visualizations**  
✅ **Plug & Play Extensibility**  
✅ **Discovery of Novel Candidates**

👥 **Target Users**  
Researchers, students, educators, biotech startups, pharmaceutical professionals, and policy makers.

🌎 **Impact & Real-World Relevance**  
- Accelerate early-stage drug target exploration  
- Enable therapeutic discovery for diseases like cancer and neurodegeneration  
- Lower barriers for smaller teams and academic labs  
- Foster collaboration and open science initiatives  

💡 **Monetization & Future Vision**  
- **Freemium Model**  
- **Consulting & Integration Services**  
- **Community Plugins/Marketplace**

⚙️ **Tech Stack**  
- **Frontend:** Next.js  
- **Backend:** FastAPI (Python)  
- **ML Models:** PyTorch, RDKit, ProtBERT  
- **LLM Integration:** Gemini AI  

⚙️ **Environment Variables**  
_Server (.env):_
\`\`\`
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
\`\`\`

_Client (.env):_
\`\`\`
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000

# Environment
NODE_ENV=development
\`\`\`

⚙️ **Installation & Setup**

**Server:**  
\`\`\`
cd server  
python -m venv venv  
.\venv\Scripts\activate  
pip install -r requirements.txt  
uvicorn main:app --reload  
\`\`\`

**Client:**  
\`\`\`
cd client  
npm install  
npm run dev  
\`\`\`

🔭 **Scalability & Future Scope**  
- Integrate more protein databases and real-time structure predictions  
- Incorporate advanced molecular docking and simulation tools  
- Expand LLM features for automated report generation  
- Add multi-target drug screening capabilities  
- Build collaborative workspaces for sharing results and workflows  

🤝 **Contributing**  
We welcome contributions! If you’d like to improve OccolusAI or add new features, feel free to open an issue or submit a pull request.

👥 **Team Members**  
- Gauri Madan
- Ashish K. Choudhary  
- Mohit Taneja  
