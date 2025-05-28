interface Candidates {
  name: string;
  smiles: string;
  score: number;
  image_base64: string
}

interface ProteinDrugDiscoveryResult {
  uniprot_id: string;
  top_candidates: Candidates[];
  insights: string;
}