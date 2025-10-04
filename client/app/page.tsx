"use client";

import { useState, useRef } from "react";
import { Search, Dna, Database, ArrowRight, FlaskConical, Loader2, Microscope, Atom, Beaker, ScrollText, Scale, Droplets, Gauge, Pill, Binary } from "lucide-react";
import { motion } from "framer-motion";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ReactMarkdown from 'react-markdown'
import { Drawer, DrawerContent, DrawerHeader, DrawerTitle, DrawerClose } from '@/components/ui/drawer';

export default function Home() {
  const [query, setQuery] = useState("");
  const [protein, setProtein] = useState<ProteinDetails>();
  const [drugDiscoveryResult, setDrugDiscoveryResult] = useState<DrugDiscoveryResult | undefined>();
  const [proteinDrugDiscoveryResult, setProteinDrugDiscoveryResult] = useState<ProteinDrugDiscoveryResult | undefined>();
  const [loading, setLoading] = useState(false);
  const [initiatingDiscovery, setInitiatingDiscovery] = useState(false);
  const [searchMode, setSearchMode] = useState<'protein' | 'drug'>('protein');
  const resultsRef = useRef<HTMLDivElement | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerLoading, setDrawerLoading] = useState(false);
  const [showResearch, setShowResearch] = useState(false);
  const [step, setStep] = useState<'search' | 'discovery' | 'done'>('search');
  const [split, setSplit] = useState(false);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL;

  const fetchData = async () => {
    if (!query) return;
    setShowResearch(true);
    setDrawerLoading(true);
    setLoading(true);
    setStep('search');
    setSplit(true);
    try {
      const response = await fetch(`https://rest.uniprot.org/uniprotkb/search?query=${query}`);
      const data = await response.json();
      if (data.results && data.results.length > 0) {
        setProtein(data.results[0]);
      }
    } catch (error) {
      console.error("Error fetching protein data:", error);
    }
    setLoading(false);
    setDrawerLoading(false);
  };

  const handleDrugDiscovery = async () => {
    setInitiatingDiscovery(true);
    setStep('discovery');
    try {
      const response = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          uniprot_id: protein?.primaryAccession,
          drug_name: query
        })
      });
      const data = await response.json();
      setDrugDiscoveryResult(data);
      setStep('done');
    } catch (error) {
      console.error("Error in drug discovery:", error);
    }
    setInitiatingDiscovery(false);
  };

  const handleProteinDrugDiscovery = async () => {
    setInitiatingDiscovery(true);
    setStep('discovery');
    try {
      console.log("🔍 Starting discovery for protein:", protein?.primaryAccession);
      const response = await fetch(`${apiUrl}/discover`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          uniprot_id: protein?.primaryAccession,
          drug_name: query,
          top_n: 5
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log("🔍 Discovery Response:", data);
      console.log("🔍 Top Candidates Count:", data.top_candidates?.length);
      
      if (data.top_candidates && data.top_candidates.length > 0) {
        console.log("🔍 First candidate:", data.top_candidates[0].name);
        console.log("🔍 First candidate has image_base64:", !!data.top_candidates[0].image_base64);
        if (data.top_candidates[0].image_base64) {
          console.log("🔍 First candidate image_base64 length:", data.top_candidates[0].image_base64.length);
          console.log("🔍 First 50 chars:", data.top_candidates[0].image_base64.substring(0, 50));
        }
      } else {
        console.warn("⚠️ No top_candidates in response!");
      }
      
      setProteinDrugDiscoveryResult(data);
      setStep('done');
    } catch (error) {
      console.error("❌ Error in drug discovery:", error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      alert(`Error: ${errorMessage}`);
      setStep('search');
    }
    setInitiatingDiscovery(false);
  };

  const renderResults = () => {
    console.log(searchMode)
    if (searchMode == 'protein' && !proteinDrugDiscoveryResult) return null;
    if(searchMode == 'drug' && !drugDiscoveryResult) return null

    if (searchMode === 'protein') {
      return (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-6">
            {proteinDrugDiscoveryResult?.top_candidates.map((candidate, index) => (
              <div key={index} className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold mb-4">Candidate: {candidate.name}</h3>

                  <div className="bg-muted/50 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-4">
                      <Binary className="h-4 w-4 text-primary" />
                      <span className="text-sm font-medium">Drug-likeness Score</span>
                    </div>
                    <div className="relative h-2 bg-primary/20 rounded-full overflow-hidden">
                      <div
                        className="absolute top-0 left-0 h-full bg-primary transition-all duration-1000 ease-out"
                        style={{ width: `${candidate.score * 100}%` }} />
                    </div>
                    <p className="mt-2 text-right text-sm font-medium">
                      {(candidate.score * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-4">Molecular Structure</h3>
                  <div className="bg-white rounded-lg p-4 flex items-center justify-center">
                    <img
                      src={`data:image/png;base64,${candidate.image_base64}`}
                      alt={`Structure of ${candidate.name}`}
                      className="max-w-full h-auto" />
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-2">SMILES Notation</h3>
                  <div className="space-y-2">
                    <p
                      className="text-sm bg-muted/50 p-3 rounded-lg font-mono break-all"
                    >
                      {proteinDrugDiscoveryResult.top_candidates[index].smiles}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div>
            <h3 className="text-2xl font-semibold mb-2 mt-8">Insights</h3>
            <div className="space-y-2">
              <ReactMarkdown>
                {proteinDrugDiscoveryResult?.insights}
              </ReactMarkdown>
            </div>
          </div>
        </>
      );
    } else {
      return (
        <>
          <div className="flex items-center gap-4 mb-6">
            <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
              <FlaskConical className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h2 className="text-2xl font-bold">Drug Discovery Results</h2>
              <p className="text-muted-foreground">Potential drug candidate for {protein?.proteinDescription?.recommendedName?.fullName?.value}</p>
            </div>
          </div><div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-6">
            <div>
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold mb-4">Drug Properties</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-muted/50 rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Scale className="h-4 w-4 text-primary" />
                        <span className="text-sm font-medium">Molecular Weight</span>
                      </div>
                      <p className="text-2xl font-bold">{drugDiscoveryResult?.molecular_weight}</p>
                      <p className="text-xs text-muted-foreground">Daltons</p>
                    </div>
                    <div className="bg-muted/50 rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Droplets className="h-4 w-4 text-primary" />
                        <span className="text-sm font-medium">LogP</span>
                      </div>
                      <p className="text-2xl font-bold">{drugDiscoveryResult?.logP}</p>
                      <p className="text-xs text-muted-foreground">Partition coefficient</p>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-4">Binding Analysis</h3>
                  <div className="bg-muted/50 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-4">
                      <Gauge className="h-4 w-4 text-primary" />
                      <span className="text-sm font-medium">Binding Probability</span>
                    </div>
                    <div className="relative h-2 bg-primary/20 rounded-full overflow-hidden">
                      <div
                        className="absolute top-0 left-0 h-full bg-primary transition-all duration-1000 ease-out"
                        style={{ width: `${(drugDiscoveryResult?.binding_probability ?? 0) * 100}%` }} />
                    </div>
                    <p className="mt-2 text-right text-sm font-medium">
                      {((drugDiscoveryResult?.binding_probability ?? 0) * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-4">Hydrogen Bonding</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-muted/50 rounded-lg p-4">
                      <p className="text-sm font-medium mb-2">H-Bond Donors</p>
                      <p className="text-2xl font-bold">{drugDiscoveryResult?.h_bond_donors}</p>
                    </div>
                    <div className="bg-muted/50 rounded-lg p-4">
                      <p className="text-sm font-medium mb-2">H-Bond Acceptors</p>
                      <p className="text-2xl font-bold">{drugDiscoveryResult?.h_bond_acceptors}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-6">
              <div className="bg-white border border-[#e5e1d8] rounded-lg p-4 flex flex-col items-center min-h-[180px]">
                <h3 className="text-lg font-semibold mb-2">Molecular Structure</h3>
                  <img
                    src={drugDiscoveryResult?.molecule_image}
                    alt="Molecular structure"
                  className="max-w-[120px] max-h-[120px] object-contain mb-2" />
                <div className="text-xs bg-muted/50 p-2 rounded font-mono break-all w-full text-center">
                  {drugDiscoveryResult?.smiles}
                </div>
              </div>
              <div className="bg-white border border-[#e5e1d8] rounded-lg p-4 flex flex-col items-center min-h-[180px]">
              <h3 className="text-lg font-semibold mb-2">Heatmap</h3>
                <img
                  src={drugDiscoveryResult?.heatmap_image}
                  alt="heatmap"
                  className="max-w-[180px] max-h-[120px] object-contain" />
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
              {(Array.isArray(drugDiscoveryResult?.top_similar_drugs) ? drugDiscoveryResult.top_similar_drugs : []).map((drug, i) => (
                <div key={i} className="bg-white border border-[#e5e1d8] rounded-lg p-4 flex flex-col items-center min-h-[180px]">
                  <h3 className="font-semibold text-base mb-1">{drug.name}</h3>
                  <p className="text-xs mb-2">Similarity: {drug.similarity}</p>
                  <img
                    src={drug.image_base64}
                    alt={`Structure of ${drug.name}`}
                    className="w-24 h-24 object-contain" />
                </div>
              ))}
            </div>
          </div>
          <div>
            <h3 className="text-2xl font-semibold mb-2 mt-8">Insights</h3>
            <div className="space-y-2">
              <ReactMarkdown>
                {drugDiscoveryResult?.insights}
              </ReactMarkdown>
            </div>
          </div>
        </>
      );
    }
  };

  return (
    <main className={`min-h-screen flex flex-row transition-all duration-500 ${split ? '' : 'bg-[#f5ecd7]'}`} style={{background: split ? undefined : '#f5ecd7'}}>
      {/* Left: Main UI (100% or 50%) */}
      <div className={`transition-all duration-500 ${split ? 'w-1/2 max-w-[50vw] border-r border-[#e5e1d8]' : 'w-full'} min-w-[400px] flex flex-col justify-center px-8 py-16`} style={{background: '#f5ecd7'}}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <div className="flex justify-center items-center mb-6 gap-6">
            {/* Hand-drawn DNA SVG Icon on the left */}
            <svg width="100" height="100" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M20 90 Q40 60 60 80 Q80 100 80 60 Q80 20 60 40 Q40 60 20 10" stroke="#111" strokeWidth="2.5" fill="none"/>
              <path d="M80 60 Q60 40 40 60 Q20 80 20 40 Q20 0 40 20 Q60 40 80 10" stroke="#111" strokeWidth="2.5" fill="none"/>
              <path d="M30 75 L70 55" stroke="#111" strokeWidth="1.5"/>
              <path d="M32 65 L68 47" stroke="#111" strokeWidth="1.5"/>
              <path d="M35 55 L65 40" stroke="#111" strokeWidth="1.5"/>
              <path d="M38 45 L62 35" stroke="#111" strokeWidth="1.5"/>
              <path d="M41 35 L59 30" stroke="#111" strokeWidth="1.5"/>
              <path d="M44 25 L56 25" stroke="#111" strokeWidth="1.5"/>
            </svg>
            {/* MOLEDRUGS title on the right */}
            <h1 className="text-[80px] font-extrabold tracking-tight" style={{letterSpacing: '-0.04em', color: '#111', fontFamily: 'Space Grotesk, Inter, Arial, sans-serif'}}>OccolusAI</h1>
          </div>
          <p className="text-2xl font-normal text-[#222] mb-2" style={{fontFamily: 'Space Grotesk'}}>An intelligent protein based drug discovery application</p>
          <p className="text-lg text-[#222] max-w-2xl mx-auto" style={{fontFamily: 'Space Grotesk'}}>
            Unlock the potential of protein-based drug discovery with our advanced search platform. Enter a protein name or UniProt ID to explore detailed molecular information and structural insights.
          </p>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="max-w-xl mx-auto mb-8"
        >
          <div className="flex flex-col items-center w-full">
            {/* Mode Switcher - Outlined pill with icons */}
            <div className="flex w-full mb-2 rounded-full border border-[#222] bg-[#faf7f2] overflow-hidden shadow-sm">
              <button
                className={`flex-1 py-2.5 px-4 text-base font-medium flex items-center justify-center gap-2 transition-colors ${searchMode === 'protein' ? 'bg-[#faf7f2] text-[#222] font-semibold' : 'bg-transparent text-[#888]'} `}
                style={{ borderRight: '1px solid #e5e1d8', borderRadius: '0' }}
                onClick={() => setSearchMode('protein')}
              >
                <Dna className="h-4 w-4" /> Protein-based
              </button>
              <button
                className={`flex-1 py-2.5 px-4 text-base font-medium flex items-center justify-center gap-2 transition-colors ${searchMode === 'drug' ? 'bg-[#faf7f2] text-[#222] font-semibold' : 'bg-transparent text-[#888]'} `}
                style={{ borderLeft: '1px solid #e5e1d8', borderRadius: '0' }}
                onClick={() => setSearchMode('drug')}
              >
                <Pill className="h-4 w-4" /> Drug-based
              </button>
            </div>
            {/* Search Bar - Outlined, rounded, subtle */}
            <div className="flex w-full mt-2">
              <input
                placeholder={searchMode === 'protein' ? 'Enter protein name or UniProt ID...' : 'Enter drug name or ID...'}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="flex-1 py-2.5 px-5 text-base border border-[#222] rounded-l-full outline-none bg-[#faf7f2] text-[#222] font-normal shadow-sm focus:ring-2 focus:ring-[#e5e1d8]"
                style={{ borderRight: 'none' }}
              />
              <button
                onClick={() => fetchData()}
                disabled={loading}
                className="py-2.5 px-6 text-base font-semibold bg-[#222] text-white rounded-r-full border border-[#222] flex items-center gap-2 shadow-sm hover:bg-[#111] transition-colors"
                style={{ borderLeft: 'none' }}
              >
                {loading ? (
                  <span className="flex items-center gap-2"><span className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-white" /> Searching...</span>
                ) : (
                  <>
                    Search
                    <ArrowRight className="ml-1 h-4 w-4" />
                  </>
                )}
              </button>
            </div>
          </div>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="grid md:grid-cols-3 gap-8 mt-16"
        >
          {/* Feature Card 1 */}
          <div className="bg-white border border-[#e5e1d8] rounded-2xl shadow-sm p-10 flex flex-col items-center text-center transition-transform hover:scale-105">
            <span className="mb-6">
              {/* Database SVG */}
              <svg width="48" height="48" fill="none" stroke="#111" strokeWidth="2.5" viewBox="0 0 48 48"><ellipse cx="24" cy="12" rx="16" ry="6"/><path d="M8 12v12c0 3.3 7.2 6 16 6s16-2.7 16-6V12"/><path d="M8 24v12c0 3.3 7.2 6 16 6s16-2.7 16-6V24"/></svg>
            </span>
            <h3 className="text-2xl font-extrabold mb-2" style={{fontFamily: 'Space Grotesk'}}>Comprehensive Database</h3>
            <p className="text-base text-[#888] font-medium">Instant access to curated, up-to-date protein and drug data from trusted sources.</p>
          </div>
          {/* Feature Card 2 */}
          <div className="bg-white border border-[#e5e1d8] rounded-2xl shadow-sm p-10 flex flex-col items-center text-center transition-transform hover:scale-105">
            <span className="mb-6">
              {/* Structure SVG */}
              <svg width="48" height="48" fill="none" stroke="#111" strokeWidth="2.5" viewBox="0 0 48 48"><path d="M12 36c8-16 16-16 24 0"/><circle cx="12" cy="36" r="3"/><circle cx="24" cy="24" r="3"/><circle cx="36" cy="36" r="3"/></svg>
            </span>
            <h3 className="text-2xl font-extrabold mb-2" style={{fontFamily: 'Space Grotesk'}}>Structural Analysis</h3>
            <p className="text-base text-[#888] font-medium">Visualize and explore protein structures and molecular properties in detail.</p>
          </div>
          {/* Feature Card 3 */}
          <div className="bg-white border border-[#e5e1d8] rounded-2xl shadow-sm p-10 flex flex-col items-center text-center transition-transform hover:scale-105">
            <span className="mb-6">
              {/* Search SVG */}
              <svg width="48" height="48" fill="none" stroke="#111" strokeWidth="2.5" viewBox="0 0 48 48"><circle cx="22" cy="22" r="10"/><path d="M34 34l-6-6"/></svg>
            </span>
            <h3 className="text-2xl font-extrabold mb-2" style={{fontFamily: 'Space Grotesk'}}>Smart Search</h3>
            <p className="text-base text-[#888] font-medium">Find proteins or drugs by name, ID, or sequence similarity with intelligent search.</p>
          </div>
        </motion.div>
                </div>
      {/* Right: Research Agent (50%) - only when split */}
      {split && (
        <div className="w-1/2 min-w-[400px] max-w-[50vw] h-screen min-h-screen overflow-y-auto bg-[#f5ecd7] px-8 py-16 border-l border-[#e5e1d8] transition-all duration-500 flex flex-col">
          {showResearch && (
            <div className="w-full max-w mx-auto flex flex-col flex-1">
              <div className="flex items-center gap-4 mb-8">
                <div className="rounded-full bg-[#f5ecd7] p-3 border border-[#e5e1d8]">
                  <svg width="48" height="48" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="18" cy="18" r="18" fill="#f5ecd7"/><ellipse cx="18" cy="14" rx="6" ry="6" fill="#fff" stroke="#111" strokeWidth="2"/><ellipse cx="18" cy="27" rx="10" ry="5" fill="#fff" stroke="#111" strokeWidth="2"/></svg>
                </div>
                <div className="text-2xl font-bold" style={{fontFamily: 'Space Grotesk'}}>Research Agent</div>
              </div>
              <div className="flex-1">
                {drawerLoading ? (
                  <div className="flex flex-col items-center justify-center h-full gap-6">
                    <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-[#111]" />
                    <div className="text-xl font-medium text-[#111]">Researching &ldquo;{query}&rdquo;...</div>
                    <div className="text-base text-[#888]">The agent is searching UniProt and analyzing results.</div>
                  </div>
                ) : protein ? (
                  <div className="bg-white rounded-2xl shadow-lg border border-[#e5e1d8] mx-auto mb-8 px-8 py-8 max-w-2xl text-base" style={{fontSize: '1rem'}}>
                    <div className="flex items-center gap-4 mb-4">
                      <div className="bg-[#f5ecd7] rounded-full p-2 border border-[#e5e1d8]">
                        <svg width="32" height="32" fill="none" stroke="#111" strokeWidth="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                  </div>
                      <div>
                      <div className="text-xl font-bold text-[#111]">{protein.proteinDescription?.recommendedName?.fullName?.value}</div>
                        <div className="flex gap-2 mt-2">
                          <span className="bg-[#f5ecd7] text-xs px-2 py-1 rounded font-semibold border border-[#e5e1d8]">UniProt ID: {protein.primaryAccession}</span>
                          <span className="bg-[#f5ecd7] text-xs px-2 py-1 rounded font-semibold border border-[#e5e1d8]">Length: {protein.sequence?.length} aa</span>
                  </div>
                </div>
              </div>
              <div className="mb-4">
                      <div className="font-semibold text-[#111] mb-1">Molecular Properties</div>
                      <div className="text-sm text-[#444]">Mass: <span className="font-medium">{protein.sequence?.molWeight ? `${(protein.sequence.molWeight / 1000).toFixed(2)} kDa` : "N/A"}</span></div>
                      <div className="text-sm text-[#444]">Organism: <span className="font-medium">{protein.organism?.scientificName || "N/A"}</span></div>
                    </div>
                    <div className="mb-4">
                      <div className="font-semibold text-[#111] mb-1">Function</div>
                      <div className="text-sm text-[#444]">{protein.comments?.find(c => c.commentType === "FUNCTION")?.texts?.[0]?.value || protein.comments?.[0]?.texts?.[0]?.value || "Functional information not available"}</div>
                    </div>
                    <div className="mb-4">
                      <div className="font-semibold text-[#111] mb-1">Gene Information</div>
                      <div className="text-sm text-[#444]">Gene: <span className="font-medium">{protein.genes?.[0]?.geneName?.value || "N/A"}</span></div>
                      <div className="text-sm text-[#444]">Alternative names: <span className="font-medium">{protein.proteinDescription?.alternativeNames?.map(name => name.fullName.value).join(", ") || "None"}</span></div>
                    </div>
                    <div className="flex justify-end mt-6">
                      <button
 className="bg-[#111] hover:bg-[#222] text-white font-bold py-3 px-6 rounded-lg text-base transition flex items-center gap-2"
                  onClick={() =>
                          searchMode === 'protein'
                            ? handleProteinDrugDiscovery()
                            : handleDrugDiscovery()
                  }
                  disabled={initiatingDiscovery}
                >
                  {initiatingDiscovery ? (
                          <span className="flex items-center gap-3"><span className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-white" /> Initiating...</span>
                  ) : (
                    <>
                            <svg width="28" height="28" fill="none" stroke="#fff" strokeWidth="2" viewBox="0 0 24 24"><path d="M6 2v6M18 2v6M4 10h16M4 10v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V10"/></svg>
                      Initiate Drug Discovery
                    </>
                  )}
                      </button>
                    </div>
                    {/* Show results if available */}
                    {step === 'done' && (
                      <div className="mt-10 space-y-10">
                        {searchMode === 'protein' && proteinDrugDiscoveryResult && (
                          <>
                            <div className="text-xl font-bold text-[#111] mb-4">Top Candidates</div>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                              {proteinDrugDiscoveryResult.top_candidates?.map((candidate, idx) => (
                                <div key={idx} className="bg-white border border-[#e5e1d8] rounded-lg p-4 flex flex-col items-center">
                                  <div className="font-bold text-lg text-[#111] mb-2">{candidate.name}</div>
                                  <div className="text-sm text-[#444] mb-1">Score: {(candidate.score * 100).toFixed(1)}%</div>
                                  <div className="text-xs text-[#888] mb-2 break-all">SMILES: {candidate.smiles}</div>
                                  {candidate.image_base64 ? (
                                    <img 
                                      src={`data:image/png;base64,${candidate.image_base64}`} 
                                      alt={candidate.name} 
                                      className="w-32 h-32 object-contain border rounded bg-white"
                                      onError={(e) => {
                                        console.error(`❌ Failed to load image for ${candidate.name}`);
                                        e.currentTarget.style.display = 'none';
                                        const parent = e.currentTarget.parentElement;
                                        if (parent) {
                                          const errorDiv = document.createElement('div');
                                          errorDiv.textContent = '⚠️ Image failed to load';
                                          errorDiv.className = 'text-red-500 text-xs';
                                          parent.appendChild(errorDiv);
                                        }
                                      }}
                                      onLoad={() => console.log(`✅ Image loaded for ${candidate.name}`)}
                                    />
                                  ) : (
                                    <div className="w-32 h-32 border rounded bg-gray-100 flex items-center justify-center text-gray-400 text-xs">
                                      No image
                                    </div>
                                  )}
                                </div>
                              ))}
                            </div>
                            {proteinDrugDiscoveryResult.insights && (
                              <div className="mt-8">
                                <div className="text-lg font-bold text-[#111] mb-2">Insights</div>
                                <div className="text-sm text-[#444] whitespace-pre-line prose max-w-none">
                                  <ReactMarkdown>{proteinDrugDiscoveryResult.insights}</ReactMarkdown>
                                </div>
                              </div>
                            )}
                          </>
                        )}
                        {searchMode === 'drug' && drugDiscoveryResult && (
                          <div className="space-y-8">
                            {/* Drug Discovery Header */}
                            <div className="flex items-center gap-4 mb-2">
                              <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                                <FlaskConical className="h-5 w-5 text-primary" />
                              </div>
                              <div>
                                <h2 className="text-2xl font-bold">Drug Discovery Results</h2>
                                <p className="text-base text-muted-foreground mt-1">Potential drug candidate for {protein?.proteinDescription?.recommendedName?.fullName?.value}</p>
                              </div>
                            </div>

                            {/* Drug Properties */}
                            <div>
                              <h3 className="text-lg font-semibold mb-3">Drug Properties</h3>
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="bg-white border border-[#e5e1d8] rounded-lg p-6 flex flex-col items-start">
                                  <div className="flex items-center gap-2 mb-2">
                                    <Scale className="h-5 w-5 text-primary" />
                                    <span className="text-base font-medium">Molecular Weight</span>
                                  </div>
                                  <p className="text-2xl font-bold">{drugDiscoveryResult?.molecular_weight}</p>
                                  <p className="text-xs text-muted-foreground">Daltons</p>
                                </div>
                                <div className="bg-white border border-[#e5e1d8] rounded-lg p-6 flex flex-col items-start">
                                  <div className="flex items-center gap-2 mb-2">
                                    <Droplets className="h-5 w-5 text-primary" />
                                    <span className="text-base font-medium">LogP</span>
                                  </div>
                                  <p className="text-2xl font-bold">{drugDiscoveryResult?.logP}</p>
                                  <p className="text-xs text-muted-foreground">Partition coefficient</p>
                                </div>
                              </div>
                            </div>

                            {/* Binding Analysis */}
                            <div>
                              <h3 className="text-lg font-semibold mb-3">Binding Analysis</h3>
                              <div className="bg-white border border-[#e5e1d8] rounded-lg p-6">
                                <div className="flex items-center gap-2 mb-2">
                                  <Gauge className="h-5 w-5 text-primary" />
                                  <span className="text-base font-medium">Binding Probability</span>
                                </div>
                                <div className="relative h-3 bg-primary/20 rounded-full overflow-hidden mb-2">
                                  <div
                                    className="absolute top-0 left-0 h-full bg-primary transition-all duration-1000 ease-out"
                                    style={{ width: `${(drugDiscoveryResult?.binding_probability ?? 0) * 100}%` }} />
                                </div>
                                <p className="text-base text-right font-medium">
                                  {((drugDiscoveryResult?.binding_probability ?? 0) * 100).toFixed(1)}%
                                </p>
                              </div>
                            </div>

                            {/* Hydrogen Bonding */}
                            <div>
                              <h3 className="text-lg font-semibold mb-3">Hydrogen Bonding</h3>
                              <div className="grid grid-cols-2 gap-6">
                                <div className="bg-white border border-[#e5e1d8] rounded-lg p-6 flex flex-col items-center">
                                  <div className="font-bold text-2xl">{drugDiscoveryResult?.h_bond_donors}</div>
                                  <div className="text-xs text-muted-foreground mt-1">H-Bond Donors</div>
                                </div>
                                <div className="bg-white border border-[#e5e1d8] rounded-lg p-6 flex flex-col items-center">
                                  <div className="font-bold text-2xl">{drugDiscoveryResult?.h_bond_acceptors}</div>
                                  <div className="text-xs text-muted-foreground mt-1">H-Bond Acceptors</div>
                                </div>
                              </div>
                            </div>

                            {/* Molecular Structure */}
                            <div className="bg-white border border-[#e5e1d8] rounded-lg p-6 flex flex-col items-center">
                              <h3 className="text-lg font-semibold mb-3">Molecular Structure</h3>
                              <img
                                src={drugDiscoveryResult?.molecule_image}
                                alt="Molecular structure"
                                className="max-w-[300px] max-h-[220px] object-contain mb-3" />
                              <div className="text-xs bg-muted/50 p-2 rounded font-mono break-all w-full text-center">
                                {drugDiscoveryResult?.smiles}
                              </div>
                            </div>

                            {/* Heatmap */}
                            <div className="bg-white border border-[#e5e1d8] rounded-lg p-6 flex flex-col items-center">
                              <h3 className="text-lg font-semibold mb-3">Heatmap</h3>
                              <img
                                src={drugDiscoveryResult?.heatmap_image}
                                alt="heatmap"
                                className="max-w-[300px] max-h-[220px] object-contain" />
                            </div>

                            {/* Similar Drugs */}
                            {Array.isArray(drugDiscoveryResult?.top_similar_drugs) && drugDiscoveryResult.top_similar_drugs.length > 0 && (
                              <div>
                                <h3 className="text-lg font-semibold mb-3">Similar Drugs</h3>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                  {drugDiscoveryResult.top_similar_drugs.map((drug, i) => (
                                    <div key={i} className="bg-white border border-[#e5e1d8] rounded-lg p-4 flex flex-col items-center min-h-[180px]">
                                      <h3 className="font-semibold text-base mb-1">{drug.name}</h3>
                                      <p className="text-xs mb-2">Similarity: {drug.similarity}</p>
                                      <img
                                        src={drug.image_base64}
                                        alt={`Structure of ${drug.name}`}
                                        className="w-24 h-24 object-contain" />
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Insights */}
                            {drugDiscoveryResult?.insights && (
                              <div className="mt-8">
                                <h3 className="text-lg font-bold text-[#111] mb-2">Insights</h3>
                                <div className="text-sm text-[#444] whitespace-pre-line prose max-w-none">
                                  <ReactMarkdown>{drugDiscoveryResult.insights}</ReactMarkdown>
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center text-[#888] text-lg">No results found.</div>
                )}
              </div>
            </div>
        )}
      </div>
      )}
    </main>
  );
}