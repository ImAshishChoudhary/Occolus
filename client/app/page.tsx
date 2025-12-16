"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Send, Loader2, ExternalLink, ArrowRight, X, Beaker, BookOpen, Lightbulb, Info, Share2, FileText, File, Box, Search, Activity } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import dynamic from "next/dynamic";

// Dynamic import for 3D viewer (client-side only)
const Molecule3DViewer = dynamic(() => import("@/components/Molecule3DViewer"), { 
  ssr: false,
  loading: () => <div className="w-full h-48 bg-[#ebe6dc] rounded-lg animate-pulse" />
});

interface Paper {
  title: string;
  summary: string;
  authors: string[];
  published: string;
  url: string;
  source: string;
}

interface ProteinInfo {
  id: string;
  name: string;
  organism: string;
  sequence_length: number;
  function: string;
}

interface ADMETData {
  drug_likeness_score?: number;
  lipinski_pass?: boolean;
  bbb_penetration?: { class: string; probability: number };
  oral_absorption?: { class: string; probability: number };
  hepatotoxicity_risk?: { class: string; risk: number };
  cardiotoxicity_risk?: { class: string; risk: number };
  tpsa?: number;
  rotatable_bonds?: number;
}

interface Analog {
  smiles: string;
  modification: string;
  molecular_weight: number;
  logP: number;
  image_base64?: string;
}

interface Candidate {
  name: string;
  smiles: string;
  score: number;
  image_base64: string;
  molecular_weight?: string;
  logP?: string;
  admet?: ADMETData;
  analogs?: Analog[];
}

interface ResearchResult {
  query: string;
  papers: Paper[];
  protein_info?: ProteinInfo;
  insights: string;
  top_candidates?: Candidate[];
  tools_used?: string[];
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState<'landing' | 'research'>('landing');
  const [result, setResult] = useState<ResearchResult | null>(null);
  const [selectedDrug, setSelectedDrug] = useState<Candidate | null>(null);
  const [drugDetails, setDrugDetails] = useState<{ admet?: ADMETData; analogs?: Analog[]; mol3d?: string; loading: boolean }>({ loading: false });
  const [displayedText, setDisplayedText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [showDesignModal, setShowDesignModal] = useState(false);
  const [novelDrugs, setNovelDrugs] = useState<Candidate[]>([]);
  const [selectedNovelDrugs, setSelectedNovelDrugs] = useState<Set<string>>(new Set());
  const [designingDrugs, setDesigningDrugs] = useState(false);
  const [show3D, setShow3D] = useState(false);
  const [showShareMenu, setShowShareMenu] = useState(false);
  const [exporting, setExporting] = useState<'pdf' | 'docx' | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Toggle novel drug selection for report
  const toggleNovelDrugSelection = (smiles: string) => {
    setSelectedNovelDrugs(prev => {
      const next = new Set(prev);
      if (next.has(smiles)) {
        next.delete(smiles);
      } else {
        next.add(smiles);
      }
      return next;
    });
  };

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  // Typing animation
  useEffect(() => {
    if (!result?.insights) {
      setDisplayedText("");
      setIsTyping(false);
      return;
    }
    
    const text = result.insights;
    setIsTyping(true);
    setDisplayedText("");
    let i = 0;
    
    const interval = setInterval(() => {
      if (i < text.length) {
        setDisplayedText(text.slice(0, i + 3));
        i += 3;
      } else {
        clearInterval(interval);
        setIsTyping(false);
      }
    }, 15);
    
    return () => clearInterval(interval);
  }, [result?.insights]);

  const handleNewSearch = () => {
    setView('landing');
    setResult(null);
    setQuery("");
    setSelectedDrug(null);
    setDrugDetails({ loading: false });
    setDisplayedText("");
  };

  // Fetch ADMET, analogs, and 3D coordinates when a drug is selected
  const fetchDrugDetails = useCallback(async (smiles: string) => {
    setDrugDetails({ loading: true });
    setShow3D(false);
    try {
      const [admetRes, analogsRes, mol3dRes] = await Promise.all([
        fetch(`${apiUrl}/admet`, {
        method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ smiles })
        }),
        fetch(`${apiUrl}/generate-analogs`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ smiles, num_analogs: 4 })
        }),
        fetch(`${apiUrl}/molecule-3d-coords`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ smiles })
        })
      ]);
      
      const admet = await admetRes.json();
      const analogs = await analogsRes.json();
      const mol3d = await mol3dRes.json();
      
      setDrugDetails({
        admet: admet.error ? undefined : admet,
        analogs: analogs.analogs || [],
        mol3d: mol3d.mol_block || undefined,
        loading: false
      });
    } catch (error) {
      console.error("Drug details error:", error);
      setDrugDetails({ loading: false });
    }
  }, [apiUrl]);

  // When a drug is selected, fetch its details
  useEffect(() => {
    if (selectedDrug?.smiles) {
      fetchDrugDetails(selectedDrug.smiles);
    }
  }, [selectedDrug, fetchDrugDetails]);

  // Export report to PDF or DOCX
  const exportReport = useCallback(async (format: 'pdf' | 'docx') => {
    if (!result) return;
    
    setExporting(format);
    try {
      const endpoint = format === 'pdf' ? '/export-pdf' : '/export-docx';
      
      // Get selected novel drugs
      const selectedNovelDrugsList = novelDrugs.filter(d => selectedNovelDrugs.has(d.smiles));
      
      const response = await fetch(`${apiUrl}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: result.query,
          insights: result.insights,
          protein_info: result.protein_info,
          top_candidates: result.top_candidates,
          novel_drugs: selectedNovelDrugsList,
          papers: result.papers
        })
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `drug_discovery_report.${format}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      } else {
        console.error("Export failed:", await response.text());
      }
    } catch (error) {
      console.error("Export error:", error);
    }
    setExporting(null);
    setShowShareMenu(false);
  }, [result, apiUrl, novelDrugs, selectedNovelDrugs]);

  // Design novel drugs using genetic algorithm
  const designNewDrugs = useCallback(async (seedSmiles?: string) => {
    setDesigningDrugs(true);
    setNovelDrugs([]);
    try {
      const response = await fetch(`${apiUrl}/design-drug`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seed_smiles: seedSmiles,
          num_generations: 30,
          population_size: 15
        })
      });
      
      const data = await response.json();
      if (data.novel_drugs) {
        setNovelDrugs(data.novel_drugs.map((d: any) => ({
          name: `Novel-${Math.random().toString(36).substr(2, 4).toUpperCase()}`,
          smiles: d.smiles,
          score: d.fitness_score,
          image_base64: d.image_base64,
          molecular_weight: d.molecular_weight?.toString(),
          logP: d.logP?.toString()
        })));
      }
    } catch (error) {
      console.error("Drug design error:", error);
    }
    setDesigningDrugs(false);
  }, [apiUrl]);

  // UNIFIED API - single call for everything
  const handleSubmit = useCallback(async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!query.trim() || loading) return;

    setView('research');
    setLoading(true);
    setResult(null);
    setDisplayedText("");
    setIsTyping(false);

    try {
      // Use the full drug discovery pipeline with ADMET
      const response = await fetch(`${apiUrl}/drug-discovery-full`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query.trim(),
          top_n: 8,
          max_papers: 15
        })
      });
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Discovery error:", error);
    }
    setLoading(false);
  }, [query, loading, apiUrl]);

  // Render formatted text - clean, consistent font size (xs = 12px)
  const renderFormattedText = (text: string) => {
    if (!text) return null;
    
    const elements: JSX.Element[] = [];
    const lines = text.split('\n');
    let key = 0;
    
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      
      // Headers ###
      if (trimmed.startsWith('### ')) {
        elements.push(
          <h3 key={key++} className="text-xs font-semibold text-[#1a1a1a] mt-5 mb-2 uppercase tracking-wide">
            {trimmed.replace('### ', '')}
          </h3>
        );
      }
      // Headers ##
      else if (trimmed.startsWith('## ')) {
        elements.push(
          <h2 key={key++} className="text-xs font-semibold text-[#1a1a1a] mt-5 mb-2">
            {trimmed.replace('## ', '')}
          </h2>
        );
      }
      // Bullet points
      else if (/^[\-\*•]\s/.test(trimmed)) {
        const content = trimmed.replace(/^[\-\*•]\s*/, '');
        const formatted = content
          .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
          .replace(/\*(.*?)\*/g, '<em>$1</em>');
        elements.push(
          <div key={key++} className="flex gap-2 mb-1.5">
            <span className="text-[#aaa] text-xs">•</span>
            <p className="text-xs text-[#444] leading-relaxed" dangerouslySetInnerHTML={{ __html: formatted }} />
                  </div>
        );
      }
      // Numbered lists
      else if (/^\d+\.\s/.test(trimmed)) {
        const match = trimmed.match(/^(\d+)\.\s*(.*)/);
        if (match) {
          const content = match[2]
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
          elements.push(
            <div key={key++} className="flex gap-2 mb-1.5">
              <span className="text-[#aaa] text-xs min-w-[16px]">{match[1]}.</span>
              <p className="text-xs text-[#444] leading-relaxed" dangerouslySetInnerHTML={{ __html: content }} />
                </div>
          );
        }
      }
      // Regular paragraph
      else {
        const formatted = trimmed
          .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
          .replace(/\*(.*?)\*/g, '<em>$1</em>');
        elements.push(
          <p key={key++} className="text-xs text-[#444] leading-relaxed mb-2" dangerouslySetInnerHTML={{ __html: formatted }} />
        );
      }
    }
    
    return elements;
  };

  // Extract key insights from the report - clean and format properly
  const extractKeyInsights = (text: string): string[] => {
    if (!text) return [];
    const insights: string[] = [];
    const lines = text.split('\n');
    
    for (const line of lines) {
      let trimmed = line.trim();
      // Skip empty lines and headers
      if (!trimmed || trimmed.startsWith('#')) continue;
      
      // Look for bullet points and important statements
      if (/^[\-\*•]\s/.test(trimmed)) {
        // Clean up the content - remove all markdown
        let content = trimmed
          .replace(/^[\-\*•]\s*/, '')  // Remove bullet prefix
          .replace(/\*\*/g, '')         // Remove bold markers
          .replace(/\*/g, '')           // Remove italic markers
          .replace(/_/g, '')            // Remove underscores
          .replace(/\[.*?\]\(.*?\)/g, '') // Remove links
          .replace(/`/g, '')            // Remove code markers
          .trim();
        
        // Only add if it's a meaningful insight
        if (content.length > 20 && content.length < 200 && !content.includes('http')) {
          insights.push(content);
        }
      }
    }
    return insights.slice(0, 5);
  };

  // Landing Page - Minimalist Drug Discovery
  if (view === 'landing') {
      return (
      <main className="min-h-screen bg-[#f5f0e8] relative overflow-hidden">
        {/* Geometric Background - Subtle Pentagon/Hexagon Lines */}
        <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="xMidYMid slice">
          <defs>
            <linearGradient id="lineGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#c8c0b4" stopOpacity="0.15" />
              <stop offset="50%" stopColor="#b8b0a4" stopOpacity="0.08" />
              <stop offset="100%" stopColor="#c8c0b4" stopOpacity="0.15" />
            </linearGradient>
            <linearGradient id="lineGrad2" x1="100%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#d4ccc0" stopOpacity="0.12" />
              <stop offset="100%" stopColor="#c0b8ac" stopOpacity="0.06" />
            </linearGradient>
          </defs>
          
          {/* Large Pentagon - Top Right */}
          <polygon 
            points="85%,5% 98%,35% 92%,65% 78%,65% 72%,35%" 
            fill="none" 
            stroke="url(#lineGrad1)" 
            strokeWidth="1"
          />
          
          {/* Medium Pentagon - Bottom Left */}
          <polygon 
            points="8%,55% 22%,48% 28%,65% 18%,80% 5%,72%" 
            fill="none" 
            stroke="url(#lineGrad2)" 
            strokeWidth="1"
          />
          
          {/* Small Pentagon - Top Left */}
          <polygon 
            points="15%,8% 25%,5% 30%,15% 22%,22% 12%,18%" 
            fill="none" 
            stroke="url(#lineGrad1)" 
            strokeWidth="0.8"
          />
          
          {/* Connecting Lines */}
          <line x1="30%" y1="15%" x2="72%" y2="35%" stroke="url(#lineGrad2)" strokeWidth="0.5" />
          <line x1="28%" y1="65%" x2="78%" y2="65%" stroke="url(#lineGrad1)" strokeWidth="0.5" />
          <line x1="22%" y1="22%" x2="8%" y2="55%" stroke="url(#lineGrad2)" strokeWidth="0.5" />
          
          {/* Accent dots at vertices */}
          <circle cx="85%" cy="5%" r="2" fill="#c8c0b4" fillOpacity="0.2" />
          <circle cx="98%" cy="35%" r="2" fill="#c8c0b4" fillOpacity="0.15" />
          <circle cx="8%" cy="55%" r="2" fill="#c8c0b4" fillOpacity="0.2" />
          <circle cx="22%" cy="48%" r="1.5" fill="#c8c0b4" fillOpacity="0.15" />
          <circle cx="15%" cy="8%" r="1.5" fill="#c8c0b4" fillOpacity="0.18" />
        </svg>

        {/* Main Content */}
        <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-6">
          <motion.div 
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className="w-full max-w-md"
          >
            {/* Logo */}
            <div className="text-center mb-10">
              <h1 className="text-3xl font-light text-[#1a1a1a] mb-2 tracking-tight">
                Occolus
              </h1>
              <p className="text-[#999] text-xs tracking-wide">Drug Discovery & Research</p>
            </div>

            {/* Search Box - Compact with integrated button */}
            <form onSubmit={handleSubmit} className="mb-6">
              <div className="flex items-center bg-white rounded-lg border border-[#ddd8ce] hover:border-[#ccc5b8] focus-within:border-[#1a1a1a] transition-colors">
                <input
                  ref={inputRef}
                  type="text"
                  placeholder="Search proteins, targets, diseases..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="flex-1 bg-transparent px-4 py-2.5 outline-none text-sm text-[#1a1a1a] placeholder-[#aaa]"
                />
                <button 
                  type="submit" 
                  disabled={!query.trim()} 
                  className="h-full px-4 py-2.5 bg-[#1a1a1a] text-white text-xs font-medium rounded-r-lg hover:bg-[#333] disabled:opacity-30 transition-colors"
                >
                  <ArrowRight className="h-4 w-4" />
                </button>
              </div>
            </form>

            {/* Example Queries */}
            <div className="flex gap-3 justify-center flex-wrap mb-12">
              {['P02533 keratin', 'EGFR inhibitors', 'Alzheimer targets'].map((ex) => (
                <button 
                  key={ex} 
                  onClick={() => setQuery(ex)} 
                  className="text-[11px] text-[#999] hover:text-[#666] transition-colors"
                >
                  {ex}
                </button>
              ))}
            </div>

            {/* Tagline */}
            <p className="text-center text-[11px] text-[#aaa] leading-relaxed max-w-xs mx-auto">
              Accelerate drug discovery with AI-powered protein analysis, compound screening, and research synthesis.
            </p>
          </motion.div>
        </div>
      </main>
    );
  }

  const keyInsights = result?.insights ? extractKeyInsights(result.insights) : [];

  // Research View
      return (
    <main className="h-screen bg-[#f5f0e8] flex flex-col overflow-hidden">
      {/* Drug Detail Modal */}
      <AnimatePresence>
        {selectedDrug && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-[#f5f0e8]/95 overflow-y-auto"
            onClick={() => setSelectedDrug(null)}
          >
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="max-w-4xl w-full my-8"
              onClick={e => e.stopPropagation()}
            >
              <button onClick={() => setSelectedDrug(null)} className="absolute top-6 right-6 p-2 text-[#666] hover:text-[#1a1a1a] bg-white/50 rounded-full">
                <X className="h-5 w-5" />
              </button>
              
              {/* Main Content */}
              <div className="flex gap-6">
                {/* Left - Structure (2D/3D toggle) */}
                <div className="flex-shrink-0 w-56">
                  {/* View Toggle */}
                  <div className="flex gap-1 mb-2">
                    <button 
                      onClick={() => setShow3D(false)}
                      className={`flex-1 py-1.5 text-[10px] rounded-lg transition ${!show3D ? 'bg-[#1a1a1a] text-white' : 'bg-white/50 text-[#888]'}`}
                    >
                      2D Structure
                    </button>
                    <button 
                      onClick={() => setShow3D(true)}
                      disabled={!drugDetails.mol3d}
                      className={`flex-1 py-1.5 text-[10px] rounded-lg transition flex items-center justify-center gap-1 ${show3D ? 'bg-[#1a1a1a] text-white' : 'bg-white/50 text-[#888]'} disabled:opacity-30`}
                    >
                      <Box className="h-3 w-3" /> 3D
                    </button>
            </div>
                  
                  {/* Structure Display */}
                  {show3D && drugDetails.mol3d ? (
                    <div className="w-56 h-56 bg-white/30 rounded-lg overflow-hidden">
                      <Molecule3DViewer molBlock={drugDetails.mol3d} width={224} height={224} />
            </div>
                  ) : selectedDrug.image_base64 ? (
                    <img src={`data:image/png;base64,${selectedDrug.image_base64}`} alt={selectedDrug.name} className="w-56 h-56 object-contain bg-white/30 rounded-lg" />
                  ) : null}
                  
                  {/* SMILES */}
                  <div className="mt-3">
                    <p className="text-[10px] text-[#888] uppercase mb-1">SMILES</p>
                    <p className="text-[10px] font-mono text-[#666] break-all bg-white/50 p-2 rounded max-w-[224px]">{selectedDrug.smiles}</p>
                      </div>
                  
                    </div>

                {/* Right - Details */}
                <div className="flex-1 min-w-0">
                  <h2 className="text-xl font-bold text-[#1a1a1a] mb-4">{selectedDrug.name}</h2>
                  
                  {/* Basic Properties */}
                  <div className="grid grid-cols-4 gap-3 mb-5">
                    <div className="bg-white/40 rounded-lg p-3 text-center">
                      <p className="text-lg font-bold text-[#1a1a1a]">{(selectedDrug.score * 100).toFixed(0)}%</p>
                      <p className="text-[9px] text-[#888]">Binding</p>
                      </div>
                    <div className="bg-white/40 rounded-lg p-3 text-center">
                      <p className="text-lg font-bold text-[#1a1a1a]">{selectedDrug.molecular_weight || 'N/A'}</p>
                      <p className="text-[9px] text-[#888]">MW (Da)</p>
                    </div>
                    <div className="bg-white/40 rounded-lg p-3 text-center">
                      <p className="text-lg font-bold text-[#1a1a1a]">{selectedDrug.logP || 'N/A'}</p>
                      <p className="text-[9px] text-[#888]">LogP</p>
                    </div>
                    <div className="bg-white/40 rounded-lg p-3 text-center">
                      <p className="text-lg font-bold text-[#1a1a1a]">{drugDetails.admet?.drug_likeness_score?.toFixed(2) || '...'}</p>
                      <p className="text-[9px] text-[#888]">Drug-likeness</p>
                  </div>
                </div>

                  {/* ADMET Properties */}
                  <div className="mb-5">
                    <h3 className="text-xs font-semibold text-[#1a1a1a] uppercase tracking-wide mb-3">ADMET Predictions</h3>
                    {drugDetails.loading ? (
                      <div className="flex items-center gap-2 text-[#888]">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span className="text-xs">Calculating ADMET...</span>
                    </div>
                    ) : drugDetails.admet ? (
                      <div className="grid grid-cols-3 gap-2">
                        <div className="bg-white/30 rounded p-2">
                          <p className="text-[10px] text-[#888]">Lipinski Rule</p>
                          <p className={`text-xs font-semibold ${drugDetails.admet.lipinski_pass ? 'text-green-600' : 'text-red-600'}`}>
                            {drugDetails.admet.lipinski_pass ? 'Pass' : 'Fail'}
                    </p>
                  </div>
                        <div className="bg-white/30 rounded p-2">
                          <p className="text-[10px] text-[#888]">BBB Penetration</p>
                          <p className="text-xs font-semibold text-[#1a1a1a]">{drugDetails.admet.bbb_penetration?.class || 'N/A'}</p>
                </div>
                        <div className="bg-white/30 rounded p-2">
                          <p className="text-[10px] text-[#888]">Oral Absorption</p>
                          <p className="text-xs font-semibold text-[#1a1a1a]">{drugDetails.admet.oral_absorption?.class || 'N/A'}</p>
                    </div>
                        <div className="bg-white/30 rounded p-2">
                          <p className="text-[10px] text-[#888]">Hepatotoxicity</p>
                          <p className={`text-xs font-semibold ${
                            drugDetails.admet.hepatotoxicity_risk?.class === 'Low' ? 'text-green-600' : 
                            drugDetails.admet.hepatotoxicity_risk?.class === 'High' ? 'text-red-600' : 'text-yellow-600'
                          }`}>{drugDetails.admet.hepatotoxicity_risk?.class || 'N/A'}</p>
                    </div>
                        <div className="bg-white/30 rounded p-2">
                          <p className="text-[10px] text-[#888]">Cardiotoxicity</p>
                          <p className={`text-xs font-semibold ${
                            drugDetails.admet.cardiotoxicity_risk?.class === 'Low' ? 'text-green-600' : 
                            drugDetails.admet.cardiotoxicity_risk?.class === 'High' ? 'text-red-600' : 'text-yellow-600'
                          }`}>{drugDetails.admet.cardiotoxicity_risk?.class || 'N/A'}</p>
                  </div>
                        <div className="bg-white/30 rounded p-2">
                          <p className="text-[10px] text-[#888]">TPSA</p>
                          <p className="text-xs font-semibold text-[#1a1a1a]">{drugDetails.admet.tpsa?.toFixed(1) || 'N/A'} Å²</p>
                </div>
              </div>
                    ) : (
                      <p className="text-xs text-[#888]">ADMET data unavailable</p>
                    )}
            </div>

                  {/* Generated Analogs */}
                <div>
                    <h3 className="text-xs font-semibold text-[#1a1a1a] uppercase tracking-wide mb-3">Generated Analogs</h3>
                    {drugDetails.loading ? (
                      <div className="flex items-center gap-2 text-[#888]">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span className="text-xs">Generating analogs...</span>
                </div>
                    ) : drugDetails.analogs && drugDetails.analogs.length > 0 ? (
                      <div className="grid grid-cols-4 gap-2">
                        {drugDetails.analogs.map((analog, i) => (
                          <div key={i} className="bg-white/30 rounded-lg p-2 text-center">
                            {analog.image_base64 && (
                              <img src={`data:image/png;base64,${analog.image_base64}`} alt={`Analog ${i+1}`} className="w-full h-16 object-contain mb-1" />
                            )}
                            <p className="text-[9px] text-[#666] truncate">{analog.modification}</p>
                            <p className="text-[10px] text-[#1a1a1a]">MW: {analog.molecular_weight}</p>
              </div>
                        ))}
              </div>
                    ) : (
                      <p className="text-xs text-[#888]">No analogs generated</p>
                    )}
            </div>
                </div>
            </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Drug Design Modal */}
      <AnimatePresence>
        {showDesignModal && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-[#f5f0e8]/95 overflow-y-auto"
            onClick={() => setShowDesignModal(false)}
          >
            <motion.div 
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="max-w-3xl w-full my-8"
              onClick={e => e.stopPropagation()}
            >
              <button onClick={() => setShowDesignModal(false)} className="absolute top-6 right-6 p-2 text-[#666] hover:text-[#1a1a1a]">
                <X className="h-5 w-5" />
              </button>

              <div className="mb-6">
                <h2 className="text-xl font-bold text-[#1a1a1a] mb-1">Generated Molecules</h2>
                <p className="text-xs text-[#888]">Select compounds to add to your research report</p>
          </div>

              {designingDrugs ? (
                <div className="flex flex-col items-center justify-center py-20">
                  <Loader2 className="h-8 w-8 animate-spin text-[#1a1a1a] mb-4" />
                  <p className="text-sm text-[#1a1a1a]">Generating molecules...</p>
                  <p className="text-xs text-[#888] mt-1">Optimizing for drug-likeness</p>
            </div>
              ) : novelDrugs.length > 0 ? (
          <div>
                  <div className="grid grid-cols-5 gap-3">
                    {novelDrugs.map((drug, i) => {
                      const isSelected = selectedNovelDrugs.has(drug.smiles);
                      return (
                        <div 
                          key={i} 
                          className="relative cursor-pointer group"
                        >
                          {/* Checkbox */}
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleNovelDrugSelection(drug.smiles);
                            }}
                            className={`absolute top-1 right-1 z-10 w-5 h-5 rounded border-2 flex items-center justify-center transition ${
                              isSelected ? 'bg-[#1a1a1a] border-[#1a1a1a]' : 'bg-white/80 border-[#ccc] hover:border-[#888]'
                            }`}
                          >
                            {isSelected && <span className="text-white text-xs">✓</span>}
                          </button>
                          
                          <div 
                            onClick={() => {
                              setSelectedDrug(drug);
                              setShowDesignModal(false);
                            }}
                            className={`bg-white/40 rounded-lg p-2 mb-2 transition ${isSelected ? 'ring-2 ring-[#1a1a1a]' : 'group-hover:bg-white/60'}`}
                          >
                            {drug.image_base64 && (
                              <img src={`data:image/png;base64,${drug.image_base64}`} alt={drug.name} className="w-full h-20 object-contain" />
                            )}
            </div>
                          <p className="text-[10px] font-medium text-[#1a1a1a] truncate">{drug.name}</p>
                          <p className="text-[9px] text-[#888]">{(drug.score * 100).toFixed(0)}% • {drug.molecular_weight}</p>
          </div>
                      );
                    })}
                    </div>
                  
                  {/* Action buttons */}
                  <div className="flex items-center justify-between mt-6 pt-4 border-t border-[#e5e0d5]">
                    <div className="text-xs text-[#888]">
                      {selectedNovelDrugs.size > 0 ? `${selectedNovelDrugs.size} selected` : 'Click checkbox to select'}
          </div>
                    <div className="flex gap-2">
              <button
                        onClick={() => designNewDrugs()}
                        className="px-3 py-1.5 text-xs text-[#888] hover:text-[#1a1a1a] transition"
              >
                        Regenerate
              </button>
              <button
                        onClick={() => setShowDesignModal(false)}
                        disabled={selectedNovelDrugs.size === 0}
                        className="px-4 py-1.5 bg-[#1a1a1a] text-white text-xs rounded-lg hover:bg-[#333] disabled:opacity-30 transition"
                      >
                        Add to Report
              </button>
            </div>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-20">
                  <Beaker className="h-10 w-10 text-[#ccc] mb-4" />
                  <p className="text-sm text-[#888] mb-4">Generate novel drug candidates</p>
              <button
                    onClick={() => designNewDrugs()}
                    className="px-5 py-2 bg-[#1a1a1a] text-white text-xs font-medium rounded-lg hover:bg-[#333] transition"
                  >
                    Generate
              </button>
            </div>
              )}
        </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Three Columns */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* LEFT - Key Insights (25%) */}
        <aside className="w-1/4 flex-shrink-0 border-r border-[#e5e0d5] flex flex-col overflow-hidden">
          <div className="px-4 py-3 border-b border-[#e5e0d5] flex items-center gap-2">
            <Lightbulb className="w-4 h-4 text-[#888]" />
            <span className="text-xs font-semibold text-[#1a1a1a] uppercase tracking-wide">Key Insights</span>
          </div>

          <div className="flex-1 overflow-y-auto scrollbar-hide p-4">
            {loading ? (
              <div className="flex flex-col items-center justify-center h-full text-[#888]">
                <Loader2 className="h-6 w-6 animate-spin mb-2" />
                <p className="text-xs">Analyzing...</p>
          </div>
            ) : result ? (
              <div className="space-y-4">
                {/* Query */}
                <div>
                  <p className="text-[10px] text-[#888] uppercase tracking-wide mb-1">Research Topic</p>
                  <p className="text-sm font-medium text-[#1a1a1a]">{result.query}</p>
          </div>

                {/* Quick Stats */}
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-[#ebe6dc]/50 rounded-lg p-2 text-center">
                    <p className="text-lg font-bold text-[#1a1a1a]">{result.papers?.length || 0}</p>
                    <p className="text-[9px] text-[#888]">Papers</p>
                </div>
                  <div className="bg-[#ebe6dc]/50 rounded-lg p-2 text-center">
                    <p className="text-lg font-bold text-[#1a1a1a]">{result.top_candidates?.length || 0}</p>
                    <p className="text-[9px] text-[#888]">Compounds</p>
                </div>
              </div>

                {/* Target Info */}
                {result.protein_info && (
                  <div>
                    <p className="text-[10px] text-[#888] uppercase tracking-wide mb-1">Target</p>
                    <p className="text-sm font-medium text-[#1a1a1a]">{result.protein_info.name}</p>
                    <p className="text-xs text-[#666]">{result.protein_info.id} • {result.protein_info.organism}</p>
                  </div>
                )}

                {/* Key Takeaways */}
                {keyInsights.length > 0 && (
                      <div>
                    <p className="text-[10px] text-[#888] uppercase tracking-wide mb-2">Key Takeaways</p>
                    <div className="space-y-2">
                      {keyInsights.map((insight, i) => (
                        <div key={i} className="flex gap-2">
                          <Info className="w-3 h-3 text-[#888] mt-0.5 flex-shrink-0" />
                          <p className="text-xs text-[#555] leading-relaxed">{insight}</p>
                  </div>
                      ))}
                </div>
              </div>
                )}

                {/* Top Compound */}
                {result.top_candidates && result.top_candidates[0] && (
                  <div>
                    <p className="text-[10px] text-[#888] uppercase tracking-wide mb-2">Top Compound</p>
                    <div 
                      className="flex items-center gap-3 p-2 bg-[#ebe6dc]/50 rounded-lg cursor-pointer hover:bg-[#ebe6dc] transition"
                      onClick={() => setSelectedDrug(result.top_candidates![0])}
                    >
                      {result.top_candidates[0].image_base64 && (
                        <img 
                          src={`data:image/png;base64,${result.top_candidates[0].image_base64}`} 
                          alt={result.top_candidates[0].name}
                          className="w-12 h-12 object-contain"
                        />
                      )}
                      <div>
                        <p className="text-xs font-medium text-[#1a1a1a]">{result.top_candidates[0].name}</p>
                        <p className="text-[10px] text-[#666]">{(result.top_candidates[0].score * 100).toFixed(1)}% match</p>
                    </div>
                    </div>
                    </div>
                )}

{/* Sources Used */}
                {result.papers && result.papers.length > 0 && (
                              <div>
                    <p className="text-[10px] text-[#888] uppercase tracking-wide mb-1">Sources</p>
                    <div className="flex flex-wrap gap-1">
                      {Array.from(new Set(result.papers.map(p => p.source))).map((src, i) => (
                        <span key={i} className={`text-[9px] px-1.5 py-0.5 rounded ${
                          src === 'arXiv' ? 'bg-orange-100 text-orange-700' : 
                          src === 'PubMed' ? 'bg-red-100 text-red-700' :
                          src === 'CrossRef' ? 'bg-purple-100 text-purple-700' :
                          'bg-gray-100 text-gray-700'
                        }`}>{src}</span>
                              ))}
                            </div>
                                </div>
                )}

                                  </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-[#999] text-center">
                <Lightbulb className="w-8 h-8 mb-2 opacity-30" />
                <p className="text-xs">Search to see key insights</p>
                              </div>
                            )}
                              </div>
        </aside>

        {/* MIDDLE - Research Report (50%) */}
        <div className="w-1/2 flex-shrink-0 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto scrollbar-hide">
            <article className="max-w-2xl mx-auto px-8 py-8">
                {loading ? (
                <div className="flex flex-col items-center justify-center h-96 text-[#888]">
                  <Loader2 className="h-8 w-8 animate-spin mb-4" />
                  <p className="text-base font-medium">Running Drug Discovery...</p>
                  <p className="text-sm text-[#aaa] mt-1">Searching papers → Analyzing protein → Predicting binding</p>
                                </div>
              ) : result ? (
                <>
                  {/* Title Section with Share */}
                  <header className="mb-8 pb-5 border-b border-[#e5e0d5]">
                    <div className="flex items-start justify-between">
                              <div>
                        <h1 className="text-2xl font-bold text-[#1a1a1a] mb-2 leading-tight">
                          {result.query}
                        </h1>
                        <p className="text-xs text-[#888]">
                          Research Report • {new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
                        </p>
                              </div>
                      
                      {/* Share Button - Top Right */}
                      <div className="relative">
                        <button
                          onClick={() => setShowShareMenu(!showShareMenu)}
                          className="flex items-center gap-2 px-3 py-1.5 text-[10px] font-medium text-[#666] bg-white/60 hover:bg-white border border-[#e5e0d5] rounded-lg transition"
                        >
                          <Share2 className="h-3 w-3" />
                          Export
                        </button>
                        
                        <AnimatePresence>
                          {showShareMenu && (
                            <motion.div
                              initial={{ opacity: 0, y: 5 }}
                              animate={{ opacity: 1, y: 0 }}
                              exit={{ opacity: 0, y: 5 }}
                              transition={{ duration: 0.15 }}
                              className="absolute top-full right-0 mt-1 bg-white rounded-lg shadow-lg border border-[#e5e0d5] overflow-hidden z-20"
                            >
                              <button
                                onClick={() => exportReport('pdf')}
                                disabled={exporting !== null}
                                className="flex items-center gap-2 w-full px-3 py-2 text-[10px] text-[#1a1a1a] hover:bg-[#f5f0e8] transition disabled:opacity-50 whitespace-nowrap"
                              >
                                {exporting === 'pdf' ? <Loader2 className="h-3 w-3 animate-spin" /> : <FileText className="h-3 w-3 text-red-500" />}
                                PDF Report
                              </button>
                              <button
                                onClick={() => exportReport('docx')}
                                disabled={exporting !== null}
                                className="flex items-center gap-2 w-full px-3 py-2 text-[10px] text-[#1a1a1a] hover:bg-[#f5f0e8] transition border-t border-[#e5e0d5] disabled:opacity-50 whitespace-nowrap"
                              >
                                {exporting === 'docx' ? <Loader2 className="h-3 w-3 animate-spin" /> : <File className="h-3 w-3 text-blue-500" />}
                                Word Document
                              </button>
                            </motion.div>
                          )}
                        </AnimatePresence>
                            </div>
                    </div>
                  </header>

                  {/* Target Protein */}
                  {result.protein_info && (
                    <section className="mb-8">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="w-1 h-5 bg-[#1a1a1a] rounded-full"></div>
                        <h2 className="text-xs font-semibold text-[#1a1a1a] uppercase tracking-wide">Target Protein</h2>
                </div>
                      <div className="bg-white/40 rounded-lg p-4">
                        <h3 className="text-sm font-semibold text-[#1a1a1a] mb-2">{result.protein_info.name}</h3>
                        <div className="grid grid-cols-3 gap-4 mb-3">
                            <div>
                            <span className="text-[10px] text-[#888]">UniProt ID</span>
                            <p className="text-xs font-medium text-[#1a1a1a]">{result.protein_info.id}</p>
                                  </div>
                          <div>
                            <span className="text-[10px] text-[#888]">Organism</span>
                            <p className="text-xs font-medium text-[#1a1a1a]">{result.protein_info.organism}</p>
                                </div>
                      <div>
                            <span className="text-[10px] text-[#888]">Length</span>
                            <p className="text-xs font-medium text-[#1a1a1a]">{result.protein_info.sequence_length} aa</p>
                                  </div>
                                </div>
                        {result.protein_info.function && (
                          <p className="text-xs text-[#555] leading-relaxed">{result.protein_info.function}</p>
                        )}
                              </div>
                    </section>
                  )}

                  {/* Compound Summary */}
                  {result.top_candidates && result.top_candidates.length > 0 && (
                    <section className="mb-8">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="w-1 h-5 bg-[#1a1a1a] rounded-full"></div>
                        <h2 className="text-xs font-semibold text-[#1a1a1a] uppercase tracking-wide">Identified Compounds</h2>
                                </div>
                      <div className="grid grid-cols-4 gap-3">
                        {result.top_candidates.slice(0, 8).map((drug, i) => (
                          <div key={i} className="text-center cursor-pointer group" onClick={() => setSelectedDrug(drug)}>
                            <div className="bg-white/40 rounded-lg p-2 mb-2 group-hover:bg-white/60 transition">
                              {drug.image_base64 && (
                                <img src={`data:image/png;base64,${drug.image_base64}`} alt={drug.name} className="w-full h-16 object-contain" />
                              )}
                                </div>
                            <p className="text-[10px] font-medium text-[#1a1a1a] truncate">{drug.name}</p>
                            <p className="text-[9px] text-[#888]">{(drug.score * 100).toFixed(0)}%</p>
                              </div>
                        ))}
                            </div>
                    </section>
                  )}

                  {/* Novel Designed Drugs - only selected ones */}
                  {selectedNovelDrugs.size > 0 && (
                    <section className="mb-8">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                          <div className="w-1 h-5 bg-[#1a1a1a] rounded-full"></div>
                          <h2 className="text-xs font-semibold text-[#1a1a1a] uppercase tracking-wide">Generated Compounds</h2>
                                </div>
                        <span className="text-[10px] text-[#888]">{selectedNovelDrugs.size} added</span>
                                </div>
                      <div className="grid grid-cols-5 gap-3">
                        {novelDrugs.filter(d => selectedNovelDrugs.has(d.smiles)).map((drug, i) => (
                          <div key={i} className="relative text-center cursor-pointer group" onClick={() => setSelectedDrug(drug)}>
                            {/* Remove button */}
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                toggleNovelDrugSelection(drug.smiles);
                              }}
                              className="absolute -top-1 -right-1 z-10 w-4 h-4 bg-[#888] hover:bg-[#666] text-white rounded-full text-[10px] flex items-center justify-center opacity-0 group-hover:opacity-100 transition"
                            >
                              ×
                            </button>
                            <div className="bg-white/40 rounded-lg p-2 mb-2 group-hover:bg-white/60 transition">
                              {drug.image_base64 && (
                                <img src={`data:image/png;base64,${drug.image_base64}`} alt={drug.name} className="w-full h-14 object-contain" />
                              )}
                              </div>
                            <p className="text-[10px] font-medium text-[#1a1a1a] truncate">{drug.name}</p>
                            <p className="text-[9px] text-[#888]">{(drug.score * 100).toFixed(0)}%</p>
                            </div>
                                  ))}
                                </div>
                    </section>
                  )}

                  {/* Analysis - with proper formatting and typing */}
                  {displayedText && (
                    <section className="mb-6">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="w-1 h-5 bg-[#1a1a1a] rounded-full"></div>
                        <h2 className="text-xs font-semibold text-[#1a1a1a] uppercase tracking-wide">Research Analysis</h2>
                              </div>
                            <div>
                        {renderFormattedText(displayedText)}
                        {isTyping && <span className="inline-block w-1 h-4 bg-[#1a1a1a] animate-pulse ml-0.5"></span>}
                            </div>
                    </section>
                  )}


                </>
              ) : null}
            </article>
                            </div>

          {/* Search Bar with New Search */}
          <div className="flex-shrink-0 px-6 py-3 border-t border-[#e5e0d5]">
            <div className="max-w-xl mx-auto flex items-center gap-3">
              <form onSubmit={handleSubmit} className="flex-1">
                <div className="flex items-center bg-white border border-[#ddd] rounded-full px-4 py-2 focus-within:border-[#1a1a1a] transition">
                  <input type="text" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Continue research..." className="flex-1 bg-transparent outline-none text-sm" />
                  <button type="submit" disabled={loading || !query.trim()} className="p-1.5 bg-[#1a1a1a] text-white rounded-full disabled:opacity-30 hover:bg-[#333]">
                    {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Send className="h-3.5 w-3.5" />}
                  </button>
                                    </div>
              </form>
              <button 
                onClick={handleNewSearch} 
                className="flex items-center gap-1.5 px-3 py-2 text-xs text-[#888] hover:text-[#1a1a1a] hover:bg-[#ebe6dc] rounded-lg transition whitespace-nowrap"
              >
                <ArrowRight className="h-3.5 w-3.5" /> New
              </button>
                                </div>
                              </div>
                            </div>

        {/* RIGHT - References (25%) */}
        <aside className="w-1/4 flex-shrink-0 border-l border-[#e5e0d5] flex flex-col overflow-hidden">
          <div className="px-4 py-3 border-b border-[#e5e0d5] flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-[#888]" />
              <span className="text-xs font-semibold text-[#1a1a1a] uppercase tracking-wide">References</span>
                                </div>
            <span className="text-[10px] text-[#888] bg-[#e5e0d5] px-2 py-0.5 rounded">{result?.papers?.length || 0}</span>
                              </div>

          <div className="flex-1 overflow-y-auto scrollbar-hide">
            {loading ? (
              <div className="flex flex-col items-center justify-center h-full text-[#888]">
                <Loader2 className="h-4 w-4 animate-spin mb-2" />
                <p className="text-xs">Finding sources...</p>
                          </div>
            ) : result?.papers?.length ? (
              <div className="divide-y divide-[#e5e0d5]/50">
                {result.papers.map((paper, i) => (
                  <a key={i} href={paper.url} target="_blank" rel="noopener noreferrer" className="flex gap-2 px-4 py-3 hover:bg-[#f0ebe0] transition">
                    <span className="text-[10px] text-[#bbb] w-4 flex-shrink-0">{i + 1}</span>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs text-[#1a1a1a] leading-snug mb-1 line-clamp-2">{paper.title}</p>
                      <p className="text-[10px] text-[#888] mb-1">{paper.authors?.[0]}{paper.authors?.length > 1 && ' et al.'}</p>
                      <div className="flex items-center gap-1.5">
                        <span className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${
                          paper.source === 'arXiv' ? 'bg-orange-100 text-orange-700' : 
                          paper.source === 'ChemRxiv' ? 'bg-green-100 text-green-700' : 
                          paper.source === 'Semantic Scholar' ? 'bg-blue-100 text-blue-700' :
                          paper.source === 'CrossRef' ? 'bg-purple-100 text-purple-700' :
                          paper.source === 'Europe PMC' ? 'bg-indigo-100 text-indigo-700' :
                          'bg-gray-100 text-gray-700'
                        }`}>{paper.source}</span>
                        <span className="text-[9px] text-[#aaa]">{paper.published}</span>
                      </div>
                    </div>
                    <ExternalLink className="h-3 w-3 text-[#ddd] flex-shrink-0" />
                  </a>
                ))}
                  </div>
                ) : (
              <div className="flex flex-col items-center justify-center h-full text-[#999]">
                <BookOpen className="w-8 h-8 mb-2 opacity-30" />
                <p className="text-xs">No sources yet</p>
            </div>
        )}
      </div>
        </aside>
      </div>

      <style jsx global>{`
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
        .line-clamp-2 { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
      `}</style>
    </main>
  );
}
