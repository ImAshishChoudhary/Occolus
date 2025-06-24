# 🔧 OccolusAI API Documentation

## Overview

The OccolusAI API provides endpoints for drug-protein interaction prediction and drug discovery. The API is built with FastAPI and provides real-time predictions using machine learning models.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API doesn't require authentication for development. For production deployment, implement API keys or OAuth.

## Rate Limiting

- **Requests per minute**: 100
- **Burst requests**: 10

## Response Format

All API responses are returned in JSON format with the following structure:

```json
{
  "status": "success|error",
  "data": {},
  "message": "Description of the response",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

```json
{
  "error": "Error description",
  "status_code": 400,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request
- `404`: Not Found
- `422`: Validation Error
- `500`: Internal Server Error

---

## Endpoints

### 1. Health Check

Check if the API is running and healthy.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "message": "Good Health",
  "status": "success",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Example:**
```bash
curl -X GET "http://localhost:8000/health"
```

---

### 2. Drug-Target Interaction Prediction

Predict the binding probability between a specific drug and protein target.

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "uniprot_id": "string",
  "drug_name": "string"
}
```

**Parameters:**
- `uniprot_id` (string, required): UniProt ID of the target protein (e.g., "P04637")
- `drug_name` (string, required): Name of the drug to analyze (e.g., "Aspirin")

**Response:**
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
  "top_similar_drugs": [
    {
      "name": "Ibuprofen",
      "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
      "similarity": 0.85,
      "image_base64": "data:image/png;base64,..."
    }
  ],
  "insights": "AI-generated analysis of the drug-target interaction..."
}
```

**Response Fields:**
- `uniprot_id`: Target protein UniProt ID
- `drug_name`: Name of the analyzed drug
- `smiles`: SMILES notation of the drug structure
- `molecular_weight`: Molecular weight in Daltons
- `logP`: Partition coefficient (lipophilicity)
- `h_bond_donors`: Number of hydrogen bond donors
- `h_bond_acceptors`: Number of hydrogen bond acceptors
- `binding_probability`: Predicted binding probability (0-1)
- `molecule_image`: Base64 encoded 2D molecular structure
- `heatmap_image`: Base64 encoded interaction heatmap
- `top_similar_drugs`: Array of structurally similar drugs
- `insights`: AI-generated analysis and recommendations

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "uniprot_id": "P04637",
    "drug_name": "Aspirin"
  }'
```

**Error Responses:**
```json
{
  "error": "Invalid UniProt ID or sequence not found",
  "status_code": 400
}
```

```json
{
  "error": "Invalid drug name or properties not found",
  "status_code": 400
}
```

---

### 3. Drug Discovery

Discover potential drug candidates for a specific protein target by screening the drug database.

**Endpoint:** `POST /discover`

**Request Body:**
```json
{
  "uniprot_id": "string",
  "top_n": 5
}
```

**Parameters:**
- `uniprot_id` (string, required): UniProt ID of the target protein
- `top_n` (integer, optional): Number of top candidates to return (default: 5, max: 30)

**Response:**
```json
{
  "uniprot_id": "P04637",
  "top_candidates": [
    {
      "name": "Aspirin",
      "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
      "score": 0.85,
      "image_base64": "data:image/png;base64,...",
      "molecular_weight": 180.16,
      "logP": 1.43,
      "h_bond_donors": 1,
      "h_bond_acceptors": 4
    },
    {
      "name": "Ibuprofen",
      "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
      "score": 0.72,
      "image_base64": "data:image/png;base64,...",
      "molecular_weight": 206.29,
      "logP": 3.97,
      "h_bond_donors": 1,
      "h_bond_acceptors": 2
    }
  ],
  "insights": "AI-generated analysis of the top candidates..."
}
```

**Response Fields:**
- `uniprot_id`: Target protein UniProt ID
- `top_candidates`: Array of drug candidates ranked by binding score
- `insights`: AI-generated analysis of the results

**Example:**
```bash
curl -X POST "http://localhost:8000/discover" \
  -H "Content-Type: application/json" \
  -d '{
    "uniprot_id": "P04637",
    "top_n": 5
  }'
```

---

## Data Models

### Protein Information

Proteins are identified by their UniProt ID and contain the following information:

```json
{
  "uniprot_id": "P04637",
  "sequence": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLYPEYLEDRQTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD",
  "description": "Cellular tumor antigen p53",
  "organism": "Homo sapiens"
}
```

### Drug Information

Drugs contain molecular properties and structural information:

```json
{
  "name": "Aspirin",
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "molecular_weight": 180.16,
  "logP": 1.43,
  "h_bond_donors": 1,
  "h_bond_acceptors": 4,
  "image_base64": "data:image/png;base64,..."
}
```

### Binding Prediction

Binding predictions include probability scores and analysis:

```json
{
  "binding_probability": 0.75,
  "confidence": "high",
  "interpretation": "Strong binding predicted",
  "recommendations": ["Consider for further testing", "Check ADMET properties"]
}
```

---

## Usage Examples

### Python Example

```python
import requests
import json

# Base URL
base_url = "http://localhost:8000"

# Health check
response = requests.get(f"{base_url}/health")
print("Health:", response.json())

# Drug-target prediction
prediction_data = {
    "uniprot_id": "P04637",
    "drug_name": "Aspirin"
}

response = requests.post(
    f"{base_url}/predict",
    json=prediction_data,
    headers={"Content-Type": "application/json"}
)

if response.status_code == 200:
    result = response.json()
    print(f"Binding probability: {result['binding_probability']:.2f}")
    print(f"Molecular weight: {result['molecular_weight']}")
else:
    print(f"Error: {response.json()['error']}")

# Drug discovery
discovery_data = {
    "uniprot_id": "P04637",
    "top_n": 5
}

response = requests.post(
    f"{base_url}/discover",
    json=discovery_data,
    headers={"Content-Type": "application/json"}
)

if response.status_code == 200:
    result = response.json()
    for candidate in result['top_candidates']:
        print(f"{candidate['name']}: {candidate['score']:.3f}")
```

### JavaScript Example

```javascript
const baseUrl = 'http://localhost:8000';

// Health check
async function checkHealth() {
    try {
        const response = await fetch(`${baseUrl}/health`);
        const data = await response.json();
        console.log('Health:', data);
    } catch (error) {
        console.error('Error:', error);
    }
}

// Drug-target prediction
async function predictInteraction(uniprotId, drugName) {
    try {
        const response = await fetch(`${baseUrl}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                uniprot_id: uniprotId,
                drug_name: drugName
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            console.log(`Binding probability: ${result.binding_probability.toFixed(2)}`);
            console.log(`Molecular weight: ${result.molecular_weight}`);
        } else {
            console.error(`Error: ${result.error}`);
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

// Drug discovery
async function discoverDrugs(uniprotId, topN = 5) {
    try {
        const response = await fetch(`${baseUrl}/discover`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                uniprot_id: uniprotId,
                top_n: topN
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            result.top_candidates.forEach(candidate => {
                console.log(`${candidate.name}: ${candidate.score.toFixed(3)}`);
            });
        } else {
            console.error(`Error: ${result.error}`);
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

// Usage
checkHealth();
predictInteraction('P04637', 'Aspirin');
discoverDrugs('P04637', 5);
```

---

## Best Practices

### 1. Error Handling
Always check for errors in API responses:

```python
response = requests.post(url, json=data)
if response.status_code == 200:
    result = response.json()
    # Process successful response
else:
    error = response.json()
    print(f"Error: {error['error']}")
```

### 2. Rate Limiting
Respect the API rate limits to avoid being throttled:

```python
import time

# Add delay between requests
time.sleep(0.1)  # 100ms delay
```

### 3. Input Validation
Validate inputs before sending requests:

```python
def validate_uniprot_id(uniprot_id):
    # UniProt IDs are typically 6-10 characters
    if not uniprot_id or len(uniprot_id) < 6:
        raise ValueError("Invalid UniProt ID")
    return uniprot_id
```

### 4. Caching
Cache frequently requested data to improve performance:

```python
import hashlib
import json

def cache_key(data):
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
```

---

## Troubleshooting

### Common Issues

1. **CORS Errors**
   - Ensure the frontend URL is in the allowed origins
   - Check CORS configuration in the backend

2. **Timeout Errors**
   - Increase request timeout for large predictions
   - Check server performance and resources

3. **Memory Issues**
   - Monitor server memory usage
   - Consider implementing result caching

4. **Model Loading Errors**
   - Ensure the model file exists
   - Check model file permissions

### Debug Mode

Enable debug mode for detailed error information:

```python
# Set environment variable
os.environ['DEBUG'] = 'True'
```

---

## Support

For API support and questions:

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/occolus-ai/issues)
- **Documentation**: [API Docs](http://localhost:8000/docs)
- **Email**: api-support@occolus-ai.com

---

## Version History

- **v1.0.0**: Initial release with basic prediction and discovery endpoints
- **v1.1.0**: Added AI insights and molecular visualization
- **v1.2.0**: Enhanced error handling and performance optimizations 