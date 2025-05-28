import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rdkit import Chem
import requests
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import numpy as np

def fetch_protein_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        lines = response.text.split("\n")
        return "".join(lines[1:])
    return None

def fetch_drug_smiles(drug_name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/CanonicalSMILES/TXT"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text.strip()
    return None

def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol=mol)
    return np.array(fp)

def encode_protein(protein, max_len=20):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    encoding = np.zeros((max_len, len(amino_acids)))
    for i, aa in enumerate(protein[:max_len]):
        if aa in amino_acids:
            encoding[i, amino_acids.index(aa)] = 1
    return encoding.flatten()

# X_protein = np.array([encode_protein(p) for p in sample_proteins])
# X_drug = np.array([smiles_to_fingerprint(d) for d in sample_drugs])
# y = torch.tensor(sample_labels, dtype=torch.float32)

class DrugTargetModel(nn.Module):
    def __init__(self, protein_dim, drug_dim):
        super(DrugTargetModel, self).__init__()
        self.fc1 = nn.Linear(protein_dim + drug_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, protein, drug):
        x = torch.cat((protein, drug), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    

# model = DrugTargetModel(protein_dim=X_protein.shape[1], drug_dim=X_drug.shape[1])
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# loss_fn = nn.BCELoss()

# X_protein_tensor = torch.tensor(X_protein, dtype=torch.float32)
# X_drug_tensor = torch.tensor(X_drug, dtype=torch.float32)

# epochs = 10
# for epoch in range(epochs):
#     optimizer.zero_grad()
#     outputs = model(X_protein_tensor, X_drug_tensor).squeeze()
#     loss = loss_fn(outputs, y)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# torch.save(model.state_dict(), "drug_target_model.pth")