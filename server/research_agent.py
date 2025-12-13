"""
Research Agent Module
Handles ArXiv paper search, PubMed integration, and AI-powered research analysis
"""

import os
import re
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# ArXiv API
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# PubMed API
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search ArXiv for papers related to the query
    """
    try:
        # Build search query for drug discovery / pharmacology
        search_query = f"all:{query} AND (cat:q-bio* OR cat:physics.bio-ph OR cat:cs.LG)"
        
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        response = requests.get(ARXIV_API_URL, params=params, timeout=30)
        
        if response.status_code != 200:
            logger.error(f"ArXiv API error: {response.status_code}")
            return []
        
        # Parse XML response
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        
        papers = []
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            published = entry.find("atom:published", ns)
            link = entry.find("atom:id", ns)
            
            # Get authors
            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.find("atom:name", ns)
                if name is not None:
                    authors.append(name.text)
            
            # Get categories
            categories = []
            for category in entry.findall("atom:category", ns):
                term = category.get("term")
                if term:
                    categories.append(term)
            
            paper = {
                "title": title.text.strip().replace("\n", " ") if title is not None else "N/A",
                "summary": summary.text.strip().replace("\n", " ")[:500] + "..." if summary is not None else "N/A",
                "authors": authors[:3],  # First 3 authors
                "published": published.text[:10] if published is not None else "N/A",
                "url": link.text if link is not None else "",
                "source": "arXiv",
                "categories": categories[:3]
            }
            papers.append(paper)
        
        logger.info(f"Found {len(papers)} papers from ArXiv for query: {query}")
        return papers
        
    except Exception as e:
        logger.error(f"Error searching ArXiv: {e}")
        return []


def search_pubmed(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search PubMed for papers related to the query
    """
    try:
        # Search for IDs
        search_params = {
            "db": "pubmed",
            "term": f"{query} drug discovery",
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance"
        }
        
        search_response = requests.get(PUBMED_SEARCH_URL, params=search_params, timeout=30)
        
        if search_response.status_code != 200:
            logger.error(f"PubMed search error: {search_response.status_code}")
            return []
        
        search_data = search_response.json()
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        
        if not id_list:
            return []
        
        # Fetch paper details
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
            "rettype": "abstract"
        }
        
        fetch_response = requests.get(PUBMED_FETCH_URL, params=fetch_params, timeout=30)
        
        if fetch_response.status_code != 200:
            logger.error(f"PubMed fetch error: {fetch_response.status_code}")
            return []
        
        # Parse XML
        root = ET.fromstring(fetch_response.content)
        
        papers = []
        for article in root.findall(".//PubmedArticle"):
            try:
                medline = article.find(".//MedlineCitation")
                article_data = medline.find(".//Article") if medline is not None else None
                
                if article_data is None:
                    continue
                
                # Get title
                title_elem = article_data.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else "N/A"
                
                # Get abstract
                abstract_elem = article_data.find(".//Abstract/AbstractText")
                abstract = abstract_elem.text[:500] + "..." if abstract_elem is not None and abstract_elem.text else "N/A"
                
                # Get authors
                authors = []
                for author in article_data.findall(".//Author"):
                    lastname = author.find("LastName")
                    forename = author.find("ForeName")
                    if lastname is not None:
                        name = lastname.text
                        if forename is not None:
                            name = f"{forename.text} {name}"
                        authors.append(name)
                
                # Get date
                pub_date = article_data.find(".//PubDate")
                year = pub_date.find("Year").text if pub_date is not None and pub_date.find("Year") is not None else "N/A"
                month = pub_date.find("Month").text if pub_date is not None and pub_date.find("Month") is not None else "01"
                
                # Get PMID for URL
                pmid_elem = medline.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else ""
                
                # Get journal
                journal_elem = article_data.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else "N/A"
                
                paper = {
                    "title": title,
                    "summary": abstract,
                    "authors": authors[:3],
                    "published": f"{year}-{month}",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                    "source": "PubMed",
                    "journal": journal
                }
                papers.append(paper)
                
            except Exception as e:
                logger.warning(f"Error parsing PubMed article: {e}")
                continue
        
        logger.info(f"Found {len(papers)} papers from PubMed for query: {query}")
        return papers
        
    except Exception as e:
        logger.error(f"Error searching PubMed: {e}")
        return []


def search_chemrxiv(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search ChemRxiv for chemistry preprints
    """
    try:
        # Use the ChemRxiv API
        url = f"https://chemrxiv.org/engage/chemrxiv/public-api/v1/items?term={query}&limit={max_results}"
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        papers = []
        
        for item in data.get("itemHits", [])[:max_results]:
            item_data = item.get("item", {})
            authors = [a.get("firstName", "") + " " + a.get("lastName", "") for a in item_data.get("authors", [])]
            
            paper = {
                "title": item_data.get("title", "N/A"),
                "summary": item_data.get("abstract", "")[:500] + "..." if item_data.get("abstract") else "N/A",
                "authors": authors[:3],
                "published": item_data.get("publishedDate", "")[:10] if item_data.get("publishedDate") else "N/A",
                "url": f"https://chemrxiv.org/engage/chemrxiv/article-details/{item_data.get('id', '')}",
                "source": "ChemRxiv",
                "journal": "ChemRxiv Preprint"
            }
            papers.append(paper)
        
        logger.info(f"Found {len(papers)} papers from ChemRxiv")
        return papers
    except Exception as e:
        logger.error(f"Error searching ChemRxiv: {e}")
        return []


def search_biorxiv(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search bioRxiv for biology preprints
    """
    try:
        # bioRxiv API - search recent papers
        url = f"https://api.biorxiv.org/details/biorxiv/2024-01-01/2025-12-31/{max_results}"
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        papers = []
        
        for item in data.get("collection", [])[:max_results]:
            # Filter by query terms
            title = item.get("title", "").lower()
            abstract = item.get("abstract", "").lower()
            query_terms = query.lower().split()
            
            if any(term in title or term in abstract for term in query_terms):
                paper = {
                    "title": item.get("title", "N/A"),
                    "summary": item.get("abstract", "")[:500] + "..." if item.get("abstract") else "N/A",
                    "authors": item.get("authors", "").split("; ")[:3],
                    "published": item.get("date", "N/A"),
                    "url": f"https://www.biorxiv.org/content/{item.get('doi', '')}",
                    "source": "bioRxiv",
                    "journal": "bioRxiv Preprint"
                }
                papers.append(paper)
        
        logger.info(f"Found {len(papers)} papers from bioRxiv")
        return papers
    except Exception as e:
        logger.error(f"Error searching bioRxiv: {e}")
        return []


def search_europe_pmc(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search Europe PMC for open access papers
    """
    try:
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {
            "query": f"{query} drug discovery",
            "format": "json",
            "pageSize": max_results,
            "sort": "RELEVANCE"
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        papers = []
        
        for item in data.get("resultList", {}).get("result", [])[:max_results]:
            authors = []
            if item.get("authorString"):
                authors = item.get("authorString", "").split(", ")[:3]
            
            paper = {
                "title": item.get("title", "N/A"),
                "summary": item.get("abstractText", "")[:500] + "..." if item.get("abstractText") else "N/A",
                "authors": authors,
                "published": item.get("firstPublicationDate", "N/A"),
                "url": f"https://europepmc.org/article/{item.get('source', 'MED')}/{item.get('id', '')}",
                "source": "Europe PMC",
                "journal": item.get("journalTitle", "N/A")
            }
            papers.append(paper)
        
        logger.info(f"Found {len(papers)} papers from Europe PMC")
        return papers
    except Exception as e:
        logger.error(f"Error searching Europe PMC: {e}")
        return []


def search_semantic_scholar(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search Semantic Scholar for academic papers
    """
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,year,abstract,url,venue"
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        papers = []
        
        for item in data.get("data", [])[:max_results]:
            authors = [a.get("name", "") for a in item.get("authors", [])]
            
            paper = {
                "title": item.get("title", "N/A"),
                "summary": item.get("abstract", "")[:500] + "..." if item.get("abstract") else "N/A",
                "authors": authors[:3],
                "published": str(item.get("year", "N/A")),
                "url": item.get("url", ""),
                "source": "Semantic Scholar",
                "journal": item.get("venue", "N/A")
            }
            papers.append(paper)
        
        logger.info(f"Found {len(papers)} papers from Semantic Scholar")
        return papers
    except Exception as e:
        logger.error(f"Error searching Semantic Scholar: {e}")
        return []


def search_crossref(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search CrossRef for academic papers
    """
    try:
        url = f"https://api.crossref.org/works"
        params = {
            "query": query,
            "rows": max_results,
            "filter": "type:journal-article"
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        papers = []
        
        for item in data.get("message", {}).get("items", [])[:max_results]:
            authors = []
            for author in item.get("author", [])[:3]:
                name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                if name:
                    authors.append(name)
            
            # Get publication date
            pub_date = item.get("published-print", item.get("published-online", {}))
            date_parts = pub_date.get("date-parts", [[]])[0]
            year = str(date_parts[0]) if date_parts else "N/A"
            
            paper = {
                "title": item.get("title", ["N/A"])[0] if item.get("title") else "N/A",
                "summary": item.get("abstract", "")[:500] if item.get("abstract") else "N/A",
                "authors": authors,
                "published": year,
                "url": item.get("URL", ""),
                "source": "CrossRef",
                "journal": item.get("container-title", ["N/A"])[0] if item.get("container-title") else "N/A"
            }
            papers.append(paper)
        
        logger.info(f"Found {len(papers)} papers from CrossRef")
        return papers
    except Exception as e:
        logger.error(f"Error searching CrossRef: {e}")
        return []


def search_research_papers(query: str, max_results: int = 20) -> List[Dict]:
    """
    Search multiple sources: ArXiv, PubMed, ChemRxiv, Semantic Scholar, CrossRef, Europe PMC
    """
    # Distribute searches across sources
    per_source = max(3, max_results // 6)
    
    # Search all sources in parallel would be better but for simplicity sequential
    arxiv_papers = search_arxiv(query, per_source)
    pubmed_papers = search_pubmed(query, per_source)
    chemrxiv_papers = search_chemrxiv(query, per_source)
    semantic_papers = search_semantic_scholar(query, per_source)
    crossref_papers = search_crossref(query, per_source)
    europepmc_papers = search_europe_pmc(query, per_source)
    
    # Combine and interleave sources for variety
    combined = []
    sources = [pubmed_papers, semantic_papers, arxiv_papers, crossref_papers, chemrxiv_papers, europepmc_papers]
    max_len = max(len(s) for s in sources) if sources else 0
    
    for i in range(max_len):
        for source in sources:
            if i < len(source):
                combined.append(source[i])
    
    logger.info(f"Total papers found: {len(combined)} from 6 sources")
    return combined[:max_results]


def get_protein_structure_url(uniprot_id: str) -> Optional[str]:
    """
    Get AlphaFold structure URL for a protein
    """
    return f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"


def get_drugbank_info(drug_name: str) -> Optional[Dict]:
    """
    Get drug information from DrugBank (mock for now, would need API key)
    """
    # This would integrate with DrugBank API
    # For now, return mock data structure
    return {
        "name": drug_name,
        "description": f"Drug compound {drug_name}",
        "indication": "Research compound",
        "mechanism": "Under investigation"
    }


class ResearchAgent:
    """
    AI-powered research agent for drug discovery
    """
    
    def __init__(self, ai_client):
        self.ai_client = ai_client
        self.tools = {
            "search_papers": self.tool_search_papers,
            "get_protein_info": self.tool_get_protein_info,
            "get_drug_info": self.tool_get_drug_info,
            "analyze_binding": self.tool_analyze_binding,
        }
    
    def tool_search_papers(self, query: str) -> Dict:
        """Search for research papers"""
        papers = search_research_papers(query, max_results=8)
        return {"papers": papers, "count": len(papers)}
    
    def tool_get_protein_info(self, uniprot_id: str) -> Dict:
        """Get protein information from UniProt"""
        try:
            response = requests.get(
                f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json",
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "id": uniprot_id,
                    "name": data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "N/A"),
                    "organism": data.get("organism", {}).get("scientificName", "N/A"),
                    "sequence_length": data.get("sequence", {}).get("length", 0),
                    "function": data.get("comments", [{}])[0].get("texts", [{}])[0].get("value", "N/A") if data.get("comments") else "N/A",
                    "structure_url": get_protein_structure_url(uniprot_id)
                }
        except Exception as e:
            logger.error(f"Error fetching protein info: {e}")
        return None
    
    def tool_get_drug_info(self, drug_name: str) -> Dict:
        """Get drug information from PubChem"""
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/CanonicalSMILES,MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,MolecularFormula/JSON"
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
                return {
                    "name": drug_name,
                    "smiles": props.get("CanonicalSMILES", "N/A"),
                    "molecular_weight": props.get("MolecularWeight", "N/A"),
                    "formula": props.get("MolecularFormula", "N/A"),
                    "logP": props.get("XLogP", "N/A"),
                    "h_bond_donors": props.get("HBondDonorCount", "N/A"),
                    "h_bond_acceptors": props.get("HBondAcceptorCount", "N/A")
                }
        except Exception as e:
            logger.error(f"Error fetching drug info: {e}")
        return None
    
    def tool_analyze_binding(self, protein_id: str, drug_name: str) -> Dict:
        """Analyze protein-drug binding potential"""
        # This would call the ML model
        return {
            "protein": protein_id,
            "drug": drug_name,
            "analysis": "pending"
        }
    
    def process_query(self, query: str, context: Dict = None) -> Dict:
        """
        Process a research query and return structured results
        """
        logger.info(f"Processing research query: {query}")
        
        # Determine query type and extract entities
        query_lower = query.lower()
        
        results = {
            "query": query,
            "papers": [],
            "protein_info": None,
            "drug_info": None,
            "insights": "",
            "tools_used": []
        }
        
        # Search for papers
        papers = search_research_papers(query, max_results=8)
        results["papers"] = papers
        results["tools_used"].append("search_papers")
        
        # Extract protein ID if mentioned
        uniprot_pattern = r'\b([A-Z][0-9][A-Z0-9]{3}[0-9])\b'
        uniprot_matches = re.findall(uniprot_pattern, query)
        
        if uniprot_matches:
            protein_info = self.tool_get_protein_info(uniprot_matches[0])
            if protein_info:
                results["protein_info"] = protein_info
                results["tools_used"].append("get_protein_info")
        
        # Check for drug names (simple heuristic)
        common_drug_suffixes = ['mab', 'nib', 'vir', 'pril', 'sartan', 'statin', 'olol', 'pine', 'zole']
        words = query.split()
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word.lower())
            if any(word_clean.endswith(suffix) for suffix in common_drug_suffixes) or len(word_clean) > 5:
                drug_info = self.tool_get_drug_info(word_clean)
                if drug_info and drug_info.get("smiles") != "N/A":
                    results["drug_info"] = drug_info
                    results["tools_used"].append("get_drug_info")
                    break
        
        return results


def build_research_prompt(query: str, papers: List[Dict], protein_info: Dict = None, drug_info: Dict = None) -> str:
    """
    Build a research paper-style prompt for AI analysis
    """
    prompt = f"""You are writing a research brief on: {query}

"""
    
    if protein_info:
        prompt += f"""Target Protein: {protein_info.get('name', 'Unknown')}
UniProt ID: {protein_info.get('id', 'N/A')} | Organism: {protein_info.get('organism', 'N/A')}
Function: {protein_info.get('function', 'Unknown function')[:300]}

"""
    
    if drug_info:
        prompt += f"""Reference Drug: {drug_info.get('name', 'N/A')}
Molecular Formula: {drug_info.get('formula', 'N/A')} | Molecular Weight: {drug_info.get('molecular_weight', 'N/A')}

"""
    
    if papers:
        prompt += "Literature Sources:\n"
        for i, paper in enumerate(papers[:6], 1):
            prompt += f"- {paper['title'][:120]} ({paper['source']})\n"
        prompt += "\n"
    
    prompt += """Write a structured research analysis with these sections:

## Background
Brief context on the research topic (2-3 sentences).

## Key Findings
Summarize what current research shows about this target/topic. Write in flowing paragraphs, not bullet points.

## Therapeutic Potential
Discuss drug discovery opportunities and potential therapeutic applications.

## Considerations
Note any challenges, safety concerns, or research gaps.

Write in academic but accessible language. Keep each section concise (2-4 sentences). Do not use numbered lists."""
    
    return prompt

