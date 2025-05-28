interface DrugDiscoveryResult {
  uniprot_id: string;
  drug_name: string;
  smiles: string;
  molecular_weight: number;
  logP: number;
  h_bond_donors: number;
  h_bond_acceptors: number;
  binding_probability: number;
  molecule_image: string;
  heatmap_image: string;
  top_similar_drugs: DrugSimilarResults[];
  insights: string
}

interface DrugSimilarResults {
  name: string;
  similarity: string;
  image_base64: string;
}
