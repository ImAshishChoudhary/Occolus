
interface Organism {
  scientificName: string
}

interface Sequence {
  length: number;
  molWeight: number
}

interface FullName {
  value: string
}

interface ProteinNames {
  fullName: FullName
}

interface ProteinDescription {
  recommendedName: ProteinNames;
  alternativeNames: ProteinNames[]
}

interface CommentText {
  value: string
}

interface ProteinComments {
  commentType: string;
  texts: CommentText[];
}

interface GeneName {
  value: string
}

interface Genes {
  geneName: GeneName
}

interface ProteinDetails {
  primaryAccession: string;
  proteinDescription: ProteinDescription;
  sequence: Sequence;
  organism: Organism;
  comments: ProteinComments[];
  genes: Genes[]
}