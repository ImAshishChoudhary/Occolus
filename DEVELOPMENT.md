# 🛠️ OccolusAI Development Guide

## Overview

This guide provides comprehensive information for developers contributing to the OccolusAI project. It covers development setup, coding standards, testing, and deployment procedures.

## Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🏗️ Project Structure](#️-project-structure)
- [🔧 Development Setup](#-development-setup)
- [📝 Coding Standards](#-coding-standards)
- [🧪 Testing](#-testing)
- [🔍 Debugging](#-debugging)
- [📦 Building & Deployment](#-building--deployment)
- [🤝 Contributing Guidelines](#-contributing-guidelines)
- [📚 API Development](#-api-development)
- [🎨 Frontend Development](#-frontend-development)
- [🤖 ML Model Development](#-ml-model-development)

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **Git**
- **Docker** (optional, for containerized development)

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/occolus-ai.git
cd occolus-ai

# Run setup script
# Linux/macOS
chmod +x setup.sh && ./setup.sh

# Windows
setup.bat

# Or manual setup
cd server && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
cd ../client && npm install
```

---

## 🏗️ Project Structure

```
occolus-ai/
├── README.md                 # Main project documentation
├── API_DOCUMENTATION.md      # API documentation
├── DEVELOPMENT.md           # This development guide
├── setup.sh                 # Linux/macOS setup script
├── setup.bat                # Windows setup script
├── server/                  # Backend (Python/FastAPI)
│   ├── main.py             # FastAPI application entry point
│   ├── model.py            # ML model definition and training
│   ├── utils.py            # Utility functions
│   ├── discovery.py        # Drug discovery logic
│   ├── requirements.txt    # Python dependencies
│   ├── pyproject.toml      # Project configuration
│   ├── drug_target_model.pth # Trained ML model
│   ├── drug_db.csv         # Drug database
│   └── venv/               # Python virtual environment
├── client/                  # Frontend (Next.js/React)
│   ├── app/                # Next.js app directory
│   │   ├── page.tsx        # Main application page
│   │   ├── layout.tsx      # Root layout
│   │   └── globals.css     # Global styles
│   ├── components/         # React components
│   │   └── ui/             # UI components (Radix UI)
│   ├── lib/                # Utility libraries
│   ├── hooks/              # Custom React hooks
│   ├── models/             # TypeScript type definitions
│   ├── public/             # Static assets
│   ├── package.json        # Node.js dependencies
│   └── tailwind.config.ts  # Tailwind CSS configuration
└── docs/                   # Additional documentation
```

---

## 🔧 Development Setup

### Backend Development

#### Environment Setup

```bash
cd server

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # Create this file for dev tools
```

#### Development Dependencies

Create `server/requirements-dev.txt`:

```txt
# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
httpx>=0.24.0

# Linting and formatting
black>=23.0.0
flake8>=6.0.0
isort>=5.12.0
mypy>=1.0.0

# Development tools
pre-commit>=3.0.0
jupyter>=1.0.0
ipython>=8.0.0
```

#### Running the Backend

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend Development

#### Environment Setup

```bash
cd client

# Install dependencies
npm install

# Install development dependencies
npm install --save-dev @types/node @types/react @types/react-dom
```

#### Running the Frontend

```bash
# Development mode
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

---

## 📝 Coding Standards

### Python (Backend)

#### Code Style

We use **Black** for code formatting and **flake8** for linting:

```bash
# Install pre-commit hooks
pre-commit install

# Format code
black server/

# Lint code
flake8 server/

# Sort imports
isort server/
```

#### Code Style Guidelines

1. **Function and Variable Names**: Use snake_case
2. **Class Names**: Use PascalCase
3. **Constants**: Use UPPER_SNAKE_CASE
4. **Docstrings**: Use Google style docstrings
5. **Type Hints**: Always use type hints for function parameters and return values

```python
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

def predict_drug_target_interaction(
    protein_sequence: str,
    drug_smiles: str,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predict drug-target interaction probability.
    
    Args:
        protein_sequence: Amino acid sequence of the target protein
        drug_smiles: SMILES notation of the drug molecule
        model_path: Optional path to the trained model
        
    Returns:
        Dictionary containing prediction results and metadata
        
    Raises:
        ValueError: If inputs are invalid
        ModelError: If model loading fails
    """
    # Implementation here
    pass
```

#### Error Handling

```python
from fastapi import HTTPException
from typing import Union

def safe_divide(a: float, b: float) -> Union[float, None]:
    """Safely divide two numbers with error handling."""
    try:
        return a / b
    except ZeroDivisionError:
        raise HTTPException(
            status_code=400,
            detail="Division by zero is not allowed"
        )
```

### TypeScript/React (Frontend)

#### Code Style

We use **ESLint** and **Prettier** for code formatting:

```bash
# Install ESLint and Prettier
npm install --save-dev eslint prettier @typescript-eslint/parser @typescript-eslint/eslint-plugin

# Run linting
npm run lint

# Format code
npm run format
```

#### Component Guidelines

```typescript
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

interface DrugPredictionProps {
  uniprotId: string;
  drugName: string;
  onResult?: (result: PredictionResult) => void;
}

interface PredictionResult {
  bindingProbability: number;
  molecularWeight: number;
  logP: number;
}

export const DrugPrediction: React.FC<DrugPredictionProps> = ({
  uniprotId,
  drugName,
  onResult
}) => {
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const handlePrediction = async (): Promise<void> => {
    try {
      setLoading(true);
      // API call implementation
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="drug-prediction"
    >
      {/* Component JSX */}
    </motion.div>
  );
};
```

---

## 🧪 Testing

### Backend Testing

#### Unit Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=server --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run tests with verbose output
pytest -v
```

#### Test Structure

```python
# tests/test_model.py
import pytest
import torch
from server.model import DrugTargetModel, encode_protein, smiles_to_fingerprint

class TestDrugTargetModel:
    def test_model_initialization(self):
        """Test model initialization with different dimensions."""
        model = DrugTargetModel(protein_dim=400, drug_dim=1024)
        assert model is not None
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
        assert hasattr(model, 'fc3')

    def test_model_forward_pass(self):
        """Test model forward pass with dummy data."""
        model = DrugTargetModel(protein_dim=400, drug_dim=1024)
        protein_tensor = torch.randn(1, 400)
        drug_tensor = torch.randn(1, 1024)
        
        output = model(protein_tensor, drug_tensor)
        assert output.shape == (1, 1)
        assert torch.all((output >= 0) & (output <= 1))

    def test_protein_encoding(self):
        """Test protein sequence encoding."""
        sequence = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLYPEYLEDRQTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
        encoded = encode_protein(sequence)
        assert encoded.shape == (400,)
        assert encoded.dtype == np.float32

    def test_smiles_fingerprinting(self):
        """Test SMILES to fingerprint conversion."""
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        fingerprint = smiles_to_fingerprint(smiles)
        assert fingerprint.shape == (1024,)
        assert fingerprint.dtype == np.float32
        assert np.all((fingerprint >= 0) & (fingerprint <= 1))
```

#### Integration Tests

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from server.main import app

client = TestClient(app)

class TestAPIEndpoints:
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["message"] == "Good Health"

    def test_predict_endpoint(self):
        """Test drug-target prediction endpoint."""
        data = {
            "uniprot_id": "P04637",
            "drug_name": "Aspirin"
        }
        response = client.post("/predict", json=data)
        assert response.status_code == 200
        
        result = response.json()
        assert "binding_probability" in result
        assert "molecular_weight" in result
        assert "smiles" in result

    def test_discover_endpoint(self):
        """Test drug discovery endpoint."""
        data = {
            "uniprot_id": "P04637",
            "top_n": 5
        }
        response = client.post("/discover", json=data)
        assert response.status_code == 200
        
        result = response.json()
        assert "top_candidates" in result
        assert len(result["top_candidates"]) <= 5
```

### Frontend Testing

#### Unit Tests

```bash
# Run tests
npm test

# Run tests with coverage
npm test -- --coverage

# Run tests in watch mode
npm test -- --watch
```

#### Test Structure

```typescript
// __tests__/components/DrugPrediction.test.tsx
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DrugPrediction } from '../../components/DrugPrediction';

describe('DrugPrediction Component', () => {
  const mockProps = {
    uniprotId: 'P04637',
    drugName: 'Aspirin',
    onResult: jest.fn()
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders without crashing', () => {
    render(<DrugPrediction {...mockProps} />);
    expect(screen.getByText(/Drug Prediction/i)).toBeInTheDocument();
  });

  it('handles prediction button click', async () => {
    render(<DrugPrediction {...mockProps} />);
    
    const button = screen.getByRole('button', { name: /predict/i });
    fireEvent.click(button);
    
    await waitFor(() => {
      expect(screen.getByText(/loading/i)).toBeInTheDocument();
    });
  });

  it('displays error message on API failure', async () => {
    // Mock API failure
    global.fetch = jest.fn(() =>
      Promise.reject(new Error('API Error'))
    );

    render(<DrugPrediction {...mockProps} />);
    
    const button = screen.getByRole('button', { name: /predict/i });
    fireEvent.click(button);
    
    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument();
    });
  });
});
```

---

## 🔍 Debugging

### Backend Debugging

#### Logging

```python
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_function(data: Optional[dict] = None) -> None:
    """Example function with debug logging."""
    logger.debug(f"Function called with data: {data}")
    
    try:
        # Your code here
        result = process_data(data)
        logger.info(f"Processing successful: {result}")
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise
```

#### Debug Mode

```python
# main.py
import os
from fastapi import FastAPI

app = FastAPI(debug=os.getenv("DEBUG", "False").lower() == "true")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )
```

### Frontend Debugging

#### React Developer Tools

Install React Developer Tools browser extension for debugging React components.

#### Console Logging

```typescript
// Development logging
const DEBUG = process.env.NODE_ENV === 'development';

export const debugLog = (message: string, data?: any): void => {
  if (DEBUG) {
    console.log(`[DEBUG] ${message}`, data);
  }
};

// Usage
debugLog('API Response', responseData);
```

#### Error Boundaries

```typescript
import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <h2>Something went wrong.</h2>
          <details>
            <summary>Error Details</summary>
            <pre>{this.state.error?.toString()}</pre>
          </details>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
```

---

## 📦 Building & Deployment

### Backend Deployment

#### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./server
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./server:/app
    restart: unless-stopped

  frontend:
    build: ./client
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend
    restart: unless-stopped
```

### Frontend Deployment

#### Production Build

```bash
# Build the application
npm run build

# Start production server
npm start
```

#### Vercel Deployment

```json
// vercel.json
{
  "version": 2,
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/next"
    }
  ],
  "env": {
    "NEXT_PUBLIC_API_URL": "@api_url"
  }
}
```

---

## 🤝 Contributing Guidelines

### Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** following the coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Commit your changes**: `git commit -m "Add amazing feature"`
7. **Push to the branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Commit Message Convention

We use conventional commit messages:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(api): add drug similarity endpoint
fix(ui): resolve binding probability display issue
docs(readme): update installation instructions
test(model): add unit tests for protein encoding
```

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance considerations addressed
- [ ] Security considerations addressed

---

## 📚 API Development

### Adding New Endpoints

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

router = APIRouter()

class NewRequest(BaseModel):
    field1: str
    field2: int

class NewResponse(BaseModel):
    result: str
    data: List[dict]

@router.post("/new-endpoint", response_model=NewResponse)
async def new_endpoint(request: NewRequest):
    """
    New endpoint description.
    
    Args:
        request: Request model
        
    Returns:
        Response model
        
    Raises:
        HTTPException: If validation fails
    """
    try:
        # Implementation
        result = process_request(request)
        return NewResponse(result="success", data=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### Database Integration

```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Drug(Base):
    __tablename__ = "drugs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    smiles = Column(String)
    molecular_weight = Column(Float)

# Database setup
engine = create_engine("sqlite:///./drugs.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)
```

---

## 🎨 Frontend Development

### Component Development

#### Creating New Components

```typescript
// components/NewComponent.tsx
import React from 'react';
import { motion } from 'framer-motion';

interface NewComponentProps {
  title: string;
  data: any[];
  onAction?: (item: any) => void;
}

export const NewComponent: React.FC<NewComponentProps> = ({
  title,
  data,
  onAction
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="new-component"
    >
      <h2>{title}</h2>
      <div className="data-list">
        {data.map((item, index) => (
          <div key={index} className="data-item">
            {/* Item content */}
          </div>
        ))}
      </div>
    </motion.div>
  );
};
```

#### Custom Hooks

```typescript
// hooks/useApi.ts
import { useState, useEffect } from 'react';

interface UseApiOptions<T> {
  url: string;
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  body?: any;
  dependencies?: any[];
}

export function useApi<T>({ url, method = 'GET', body, dependencies = [] }: UseApiOptions<T>) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const response = await fetch(url, {
          method,
          headers: {
            'Content-Type': 'application/json',
          },
          body: body ? JSON.stringify(body) : undefined,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, dependencies);

  return { data, loading, error };
}
```

---

## 🤖 ML Model Development

### Model Training

```python
# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

def train_model(X_protein, X_drug, y, epochs=100, batch_size=32):
    """
    Train the drug-target interaction model.
    
    Args:
        X_protein: Protein feature matrix
        X_drug: Drug feature matrix
        y: Target labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained model
    """
    # Split data
    X_protein_train, X_protein_test, X_drug_train, X_drug_test, y_train, y_test = train_test_split(
        X_protein, X_drug, y, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(X_protein_train, dtype=torch.float32),
        torch.tensor(X_drug_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = DrugTargetModel(
        protein_dim=X_protein.shape[1],
        drug_dim=X_drug.shape[1]
    )
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for protein_batch, drug_batch, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(protein_batch, drug_batch).squeeze()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    return model

# Usage
if __name__ == "__main__":
    # Load your data
    X_protein = np.random.randn(1000, 400)  # Example data
    X_drug = np.random.randn(1000, 1024)    # Example data
    y = np.random.randint(0, 2, 1000)       # Example labels
    
    # Train model
    model = train_model(X_protein, X_drug, y)
    
    # Save model
    torch.save(model.state_dict(), "drug_target_model.pth")
```

### Model Evaluation

```python
# evaluate_model.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X_protein, X_drug, y):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_protein: Protein feature matrix
        X_drug: Drug feature matrix
        y: True labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    with torch.no_grad():
        protein_tensor = torch.tensor(X_protein, dtype=torch.float32)
        drug_tensor = torch.tensor(X_drug, dtype=torch.float32)
        
        predictions = model(protein_tensor, drug_tensor).squeeze().numpy()
        predicted_labels = (predictions > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(y, predicted_labels, average='binary')
    auc = roc_auc_score(y, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }

# Usage
if __name__ == "__main__":
    # Load model and test data
    model = DrugTargetModel(protein_dim=400, drug_dim=1024)
    model.load_state_dict(torch.load("drug_target_model.pth"))
    
    # Load test data
    X_protein_test = np.random.randn(200, 400)  # Example test data
    X_drug_test = np.random.randn(200, 1024)    # Example test data
    y_test = np.random.randint(0, 2, 200)       # Example test labels
    
    # Evaluate
    metrics = evaluate_model(model, X_protein_test, X_drug_test, y_test)
    
    print("Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
```

---

## 📞 Support

For development support:

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/occolus-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/occolus-ai/discussions)
- **Email**: dev-support@occolus-ai.com

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 