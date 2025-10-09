# 🧬 OccolusAI - Intelligent Protein-Based Drug Discovery Platform

<div align="center">

![OccolusAI Logo](https://img.shields.io/badge/OccolusAI-Drug%20Discovery-blue?style=for-the-badge&logo=molecule)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Next.js](https://img.shields.io/badge/Next.js-13+-black?style=for-the-badge&logo=next.js)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green?style=for-the-badge&logo=fastapi)

**Revolutionizing drug discovery through AI-powered protein-drug interaction prediction**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](http://makeapullrequest.com)

</div>

---

<img width="1838" height="899" alt="image" src="https://github.com/user-attachments/assets/370954d2-a41a-45c8-be0f-05611423b1f4" />

---

## 📋 Table of Contents

- [🎯 Problem Statement](#-problem-statement)
- [💡 Solution Overview](#-solution-overview)
- [🚀 Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [🛠️ Tech Stack](#️-tech-stack)
- [📦 Installation & Setup](#-installation--setup)
- [🎮 Usage Guide](#-usage-guide)
- [🔬 Technical Details](#-technical-details)
- [📊 Model Performance](#-model-performance)
- [🔧 API Documentation](#-api-documentation)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)


---

## 🎯 Problem Statement

Traditional drug discovery is a **costly, time-consuming, and high-risk** process:

- **💰 Cost**: $2.6 billion average cost to bring a drug to market
- **⏰ Time**: 10-15 years from discovery to approval
- **🎯 Success Rate**: Only 1 in 10,000 compounds reach clinical trials
- **🔬 Manual Process**: Heavy reliance on trial-and-error approaches
- **📊 Data Overload**: Difficulty in analyzing vast molecular datasets

**Key Challenges:**
- Predicting drug-protein interactions accurately
- Identifying drug repurposing opportunities
- Analyzing molecular properties efficiently
- Visualizing complex molecular structures
- Generating actionable insights from data

---

## 💡 Solution Overview

**OccolusAI** is an **intelligent drug discovery platform** that leverages **machine learning** and **artificial intelligence** to predict drug-protein interactions and accelerate the drug discovery process.

### 🎯 Core Capabilities

1. **🤖 AI-Powered Prediction**: Deep learning models predict drug-protein binding probabilities
2. **🔍 Intelligent Search**: Dual-mode discovery (protein-based & drug-based)
3. **📊 Molecular Analysis**: Comprehensive drug property analysis
4. **🎨 Visual Insights**: Interactive molecular structure visualization
5. **🧠 AI Insights**: Automated analysis and recommendations

---

## 🚀 Key Features

### 🔬 **Dual Discovery Modes**

#### **Protein-Based Discovery**
- Search by protein name or UniProt ID
- Screen FDA-approved drugs against target proteins
- Rank candidates by binding probability
- Generate AI-powered insights

#### **Drug-Based Analysis**
- Analyze specific drug-protein interactions
- Predict binding affinity and properties
- Visualize molecular structures
- Identify similar compounds

### 🧬 **Molecular Analysis**

- **📏 Molecular Properties**: Weight, LogP, H-bond donors/acceptors
- **🎨 Structure Visualization**: 2D molecular structure rendering
- **🔥 Interaction Heatmaps**: Feature importance visualization
- **📈 Binding Scores**: Probability-based interaction predictions

### 🤖 **AI-Powered Insights**

- **💡 Drug-likeness Assessment**: AI evaluation of compound properties
- **🔄 Repurposing Opportunities**: Identification of new therapeutic uses
- **📚 Literature Analysis**: Context from scientific literature
- **🎯 Recommendations**: Next steps for lead optimization

### 🎨 **Modern User Interface**

- **📱 Responsive Design**: Mobile-friendly interface
- **⚡ Real-time Search**: Instant protein/drug lookup
- **🖥️ Split-Screen Layout**: Research agent panel
- **🎭 Beautiful Animations**: Smooth user experience

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   External      │
│   (Next.js)     │◄──►│   (FastAPI)     │◄──►│   APIs          │
│                 │    │                 │    │                 │
│ • React UI      │    │ • ML Models     │    │ • UniProt       │
│ • TypeScript    │    │ • RDKit         │    │ • PubChem       │
│ • Tailwind CSS  │    │ • PyTorch       │    │ • Gemini AI     │
│ • Framer Motion │    │ • FastAPI       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Data Flow**

1. **User Input** → Protein/Drug search query
2. **API Integration** → Fetch molecular data from UniProt/PubChem
3. **ML Processing** → Drug-target interaction prediction
4. **AI Analysis** → Generate insights using Gemini AI
5. **Visualization** → Render molecular structures and heatmaps
6. **Results Display** → Present comprehensive analysis

---

## 🛠️ Tech Stack

### **Backend (Python)**
| Technology | Version | Purpose |
|------------|---------|---------|
| **FastAPI** | 0.115+ | High-performance web framework |
| **PyTorch** | 2.6+ | Deep learning framework |
| **RDKit** | 2024.9+ | Chemical informatics |
| **Google Gemini** | 1.9+ | AI-powered insights |
| **NumPy** | 2.2+ | Numerical computing |
| **Pandas** | 2.2+ | Data manipulation |
| **Matplotlib** | 3.10+ | Visualization |
| **Seaborn** | 0.13+ | Statistical visualization |

### **Frontend (TypeScript/React)**
| Technology | Version | Purpose |
|------------|---------|---------|
| **Next.js** | 13+ | React framework |
| **TypeScript** | 5.2+ | Type safety |
| **Tailwind CSS** | 3.3+ | Styling |
| **Framer Motion** | 11+ | Animations |
| **Radix UI** | Latest | Component library |
| **React Hook Form** | 7.5+ | Form handling |
| **React Markdown** | 10+ | Markdown rendering |

### **External APIs**
- **UniProt**: Protein sequence and annotation data
- **PubChem**: Chemical compound information
- **Google Gemini**: AI-powered analysis and insights

---

## 📦 Installation & Setup

### **Prerequisites**

- **Python 3.11+**
- **Node.js 18+**
- **Git**
- **Google Gemini API Key** (for AI insights)

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/occolus-ai.git
cd occolus-ai
```

### **2. Backend Setup**

```bash
# Navigate to server directory
cd server

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### **3. Frontend Setup**

```bash
# Navigate to client directory
cd ../client

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env.local
# Edit .env.local and add your API URL
```

### **4. Environment Variables**

#### **Backend (.env)**
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

#### **Frontend (.env.local)**
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### **5. Run the Application**

```bash
# Terminal 1: Start Backend
cd server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Frontend
cd client
npm run dev
```

### **6. Access the Application**

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

---

## 🎮 Usage Guide

### **Getting Started**

1. **Open the Application**: Navigate to http://localhost:3000
2. **Choose Mode**: Select between "Protein-based" or "Drug-based" discovery
3. **Enter Query**: 
   - **Protein Mode**: Enter protein name or UniProt ID (e.g., "P53", "P04637")
   - **Drug Mode**: Enter drug name (e.g., "Aspirin", "Ibuprofen")
4. **Analyze Results**: Review predictions, visualizations, and AI insights

### **Protein-Based Discovery**

1. **Search for Protein**: Enter protein name or UniProt ID
2. **View Protein Info**: See sequence, description, and properties
3. **Discover Drugs**: Click "Discover Candidates" to screen FDA-approved drugs
4. **Analyze Results**: Review binding scores, molecular structures, and insights

### **Drug-Based Analysis**

1. **Search for Drug**: Enter drug name
2. **Select Protein**: Choose target protein for analysis
3. **View Analysis**: See binding probability, molecular properties, and visualizations
4. **Explore Similar Drugs**: Find structurally similar compounds

### **Understanding Results**

#### **Binding Probability**
- **0.0-0.3**: Low binding affinity
- **0.3-0.7**: Moderate binding affinity
- **0.7-1.0**: High binding affinity

#### **Molecular Properties**
- **Molecular Weight**: Should be < 500 Da for good bioavailability
- **LogP**: Should be between 1-3 for optimal membrane permeability
- **H-bond Donors**: Should be ≤ 5
- **H-bond Acceptors**: Should be ≤ 10

---

## 🔬 Technical Details

### **Machine Learning Model**

#### **Architecture**
```python
class DrugTargetModel(nn.Module):
    def __init__(self, protein_dim, drug_dim):
        super(DrugTargetModel, self).__init__()
        self.fc1 = nn.Linear(protein_dim + drug_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
```

#### **Input Features**
- **Protein Encoding**: 400-dimensional vector (20 amino acids × 20 max length)
- **Drug Fingerprinting**: 1024-bit Morgan fingerprints using RDKit
- **Model Size**: 748KB trained model

#### **Training Data**
- **Drug Database**: 30 FDA-approved drugs with SMILES notation
- **Protein Targets**: Various protein sequences from UniProt
- **Model Type**: Binary classification (binding probability 0-1)

### **Data Processing Pipeline**

1. **Protein Sequence Fetching**: UniProt API integration
2. **Drug Information Retrieval**: PubChem API integration
3. **Molecular Fingerprinting**: RDKit Morgan fingerprints
4. **Feature Engineering**: Protein encoding and drug representation
5. **Model Prediction**: PyTorch inference
6. **Visualization**: Matplotlib/Seaborn heatmaps
7. **AI Analysis**: Gemini AI insights generation

### **API Endpoints**

#### **Health Check**
```http
GET /health
```

#### **Drug-Target Prediction**
```http
POST /predict
{
  "uniprot_id": "P04637",
  "drug_name": "Aspirin"
}
```

#### **Drug Discovery**
```http
POST /discover
{
  "uniprot_id": "P04637",
  "top_n": 5
}
```

---

## 📊 Model Performance

### **Current Capabilities**
- **Prediction Accuracy**: Binary classification for drug-protein binding
- **Processing Speed**: Real-time predictions (< 1 second)
- **Database Coverage**: 30 FDA-approved drugs
- **Protein Support**: Any UniProt-accessible protein

### **Model Metrics**
- **Input Dimensions**: Protein (400) + Drug (1024) = 1424 features
- **Hidden Layers**: 128 → 64 → 1 neurons
- **Activation**: ReLU + Sigmoid
- **Loss Function**: Binary Cross-Entropy

### **Limitations & Future Improvements**
- **Dataset Size**: Expand beyond 30 drugs
- **Model Architecture**: Implement Graph Neural Networks
- **Validation**: Add cross-validation and uncertainty quantification
- **3D Structure**: Integrate molecular docking

---

## 🔧 API Documentation

### **Authentication**
Currently, the API doesn't require authentication for development. For production, implement API keys or OAuth.

### **Rate Limiting**
- **Requests per minute**: 100
- **Burst requests**: 10

### **Error Handling**
```json
{
  "error": "Invalid UniProt ID or sequence not found",
  "status_code": 400
}
```

### **Response Format**
```json
{
  "uniprot_id": "P04637",
  "drug_name": "Aspirin",
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "molecular_weight": 180.16,
  "logP": 1.43,
  "h_bond_donors": 1,
  "h_bond_acceptors": 4,
  "binding_probability": 0.75,
  "molecule_image": "data:image/png;base64,...",
  "heatmap_image": "data:image/png;base64,...",
  "insights": "AI-generated analysis..."
}
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

### **1. Fork the Repository**
```bash
git clone https://github.com/yourusername/occolus-ai.git
cd occolus-ai
```

### **2. Create a Feature Branch**
```bash
git checkout -b feature/amazing-feature
```

### **3. Make Changes**
- Follow the existing code style
- Add tests for new features
- Update documentation

### **4. Commit Changes**
```bash
git commit -m "Add amazing feature"
```

### **5. Push to Branch**
```bash
git push origin feature/amazing-feature
```

### **6. Open Pull Request**
Create a pull request with a detailed description of your changes.

### **Development Guidelines**
- **Code Style**: Follow PEP 8 (Python) and ESLint (TypeScript)
- **Testing**: Add unit tests for new features
- **Documentation**: Update README and API docs
- **Commits**: Use conventional commit messages

---

## 🐛 Troubleshooting

### **Common Issues**

#### **Backend Issues**
```bash
# Module not found errors
pip install -r requirements.txt

# RDKit installation issues (Windows)
conda install -c conda-forge rdkit

# Port already in use
lsof -ti:8000 | xargs kill -9
```

#### **Frontend Issues**
```bash
# Node modules issues
rm -rf node_modules package-lock.json
npm install

# Build errors
npm run build
```

#### **API Issues**
- **CORS Errors**: Check backend CORS configuration
- **Timeout Errors**: Increase request timeout
- **Memory Issues**: Monitor system resources

### **Performance Optimization**
- **Model Caching**: Implement model result caching
- **Database Indexing**: Optimize database queries
- **CDN**: Use CDN for static assets
- **Compression**: Enable gzip compression

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **UniProt Consortium** for protein data
- **PubChem** for chemical compound information
- **Google Gemini** for AI-powered insights
- **RDKit** for chemical informatics
- **PyTorch** for deep learning capabilities
- **Next.js** for the React framework

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/occolus-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/occolus-ai/discussions)
- **Email**: support@occolus-ai.com

---

<div align="center">

**Made with ❤️ by the OccolusAI Team**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/occolus-ai?style=social)](https://github.com/yourusername/occolus-ai)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/occolus-ai?style=social)](https://github.com/yourusername/occolus-ai)
[![GitHub issues](https://img.shields.io/github/issues/yourusername/occolus-ai)](https://github.com/yourusername/occolus-ai/issues)

</div>
