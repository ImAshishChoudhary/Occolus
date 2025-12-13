"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Send, Loader2, ExternalLink, Plus, ArrowRight, X, Beaker, BookOpen, Lightbulb, Info } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

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

interface Candidate {
  name: string;
  smiles: string;
  score: number;
  image_base64: string;
  molecular_weight?: string;
  logP?: string;
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
  const [displayedText, setDisplayedText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

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
    setDisplayedText("");
  };

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
      const response = await fetch(`${apiUrl}/unified-discovery`, {
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

  // Render formatted text - clean, same font size throughout
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
          <h3 key={key++} className="text-sm font-semibold text-[#1a1a1a] mt-5 mb-2">
            {trimmed.replace('### ', '')}
          </h3>
        );
      }
      // Headers ##
      else if (trimmed.startsWith('## ')) {
        elements.push(
          <h2 key={key++} className="text-sm font-semibold text-[#1a1a1a] mt-5 mb-2">
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
            <span className="text-[#888]">•</span>
            <p className="text-sm text-[#444] leading-relaxed" dangerouslySetInnerHTML={{ __html: formatted }} />
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
              <span className="text-[#888] min-w-[16px]">{match[1]}.</span>
              <p className="text-sm text-[#444] leading-relaxed" dangerouslySetInnerHTML={{ __html: content }} />
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
          <p key={key++} className="text-sm text-[#444] leading-relaxed mb-2" dangerouslySetInnerHTML={{ __html: formatted }} />
        );
      }
    }
    
    return elements;
  };

  // Extract key insights from the report
  const extractKeyInsights = (text: string): string[] => {
    if (!text) return [];
    const insights: string[] = [];
    const lines = text.split('\n');
    
    for (const line of lines) {
      const trimmed = line.trim();
      // Look for bullet points and important statements
      if (/^[\-\*•]\s/.test(trimmed)) {
        const content = trimmed.replace(/^[\-\*•]\s*/, '').replace(/\*\*/g, '');
        if (content.length > 20 && content.length < 200) {
          insights.push(content);
        }
      }
    }
    return insights.slice(0, 6);
  };

  // Landing Page - Clean Minimalist
  if (view === 'landing') {
      return (
      <main className="h-screen bg-[#f5f0e8] flex flex-col items-center justify-center px-6">
        <div className="max-w-md w-full">
          {/* Logo */}
          <h1 className="text-3xl font-light text-[#1a1a1a] text-center mb-2">
            Occolus<span className="font-semibold">AI</span>
          </h1>
          <p className="text-[#999] text-xs text-center mb-10">Drug Discovery Platform</p>

          {/* Search */}
          <form onSubmit={handleSubmit} className="mb-6">
            <div className="flex items-center bg-white rounded-full px-4 py-3 border border-[#e5e0d5] focus-within:border-[#1a1a1a] transition">
              <input
                ref={inputRef}
                type="text"
                placeholder="Search proteins, diseases, targets..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="flex-1 bg-transparent outline-none text-sm text-[#1a1a1a] placeholder-[#bbb]"
              />
              <button 
                type="submit" 
                disabled={!query.trim()} 
                className="ml-2 bg-[#1a1a1a] text-white rounded-full p-2 hover:bg-[#333] disabled:opacity-20 transition"
              >
                <ArrowRight className="h-4 w-4" />
              </button>
                    </div>
          </form>

          {/* Examples */}
          <div className="flex gap-2 justify-center flex-wrap">
            {['P02533 keratin', 'EGFR inhibitors', 'Alzheimer drugs'].map((ex) => (
              <button 
                key={ex} 
                onClick={() => setQuery(ex)} 
                className="px-3 py-1.5 text-xs text-[#888] hover:text-[#1a1a1a] transition"
              >
                {ex}
              </button>
            ))}
          </div>
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
            className="fixed inset-0 z-50 flex items-center justify-center p-8 bg-[#f5f0e8]/95"
            onClick={() => setSelectedDrug(null)}
          >
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="max-w-2xl w-full"
              onClick={e => e.stopPropagation()}
            >
              <button onClick={() => setSelectedDrug(null)} className="absolute top-8 right-8 p-2 text-[#666] hover:text-[#1a1a1a]">
                <X className="h-6 w-6" />
              </button>
              
              <div className="flex gap-8 items-start">
                {selectedDrug.image_base64 && (
                  <img src={`data:image/png;base64,${selectedDrug.image_base64}`} alt={selectedDrug.name} className="w-72 h-72 object-contain" />
                )}
                <div className="flex-1">
                  <h2 className="text-2xl font-bold text-[#1a1a1a] mb-4">{selectedDrug.name}</h2>
                  <div className="space-y-3">
                    <div className="flex justify-between border-b border-[#e5e0d5] pb-2">
                      <span className="text-sm text-[#888]">Binding Score</span>
                      <span className="text-sm font-semibold text-[#1a1a1a]">{(selectedDrug.score * 100).toFixed(1)}%</span>
            </div>
                    {selectedDrug.molecular_weight && (
                      <div className="flex justify-between border-b border-[#e5e0d5] pb-2">
                        <span className="text-sm text-[#888]">Molecular Weight</span>
                        <span className="text-sm font-semibold text-[#1a1a1a]">{selectedDrug.molecular_weight} Da</span>
                      </div>
                    )}
                    {selectedDrug.logP && (
                      <div className="flex justify-between border-b border-[#e5e0d5] pb-2">
                        <span className="text-sm text-[#888]">LogP</span>
                        <span className="text-sm font-semibold text-[#1a1a1a]">{selectedDrug.logP}</span>
                      </div>
                    )}
                    </div>
                  <div className="mt-4">
                    <p className="text-xs text-[#888] mb-1">SMILES</p>
                    <p className="text-xs font-mono text-[#666] break-all bg-white/50 p-2 rounded">{selectedDrug.smiles}</p>
                  </div>
                </div>
              </div>
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
                  {/* Title Section */}
                  <header className="mb-8 pb-6 border-b border-[#e5e0d5]">
                    <h1 className="text-2xl font-bold text-[#1a1a1a] mb-2 leading-tight">
                      Research Report: {result.query}
                    </h1>
                    <p className="text-xs text-[#888]">
                      Generated {new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
                    </p>
                  </header>

                  {/* Target Protein */}
                  {result.protein_info && (
                    <section className="mb-8">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="w-1 h-5 bg-[#1a1a1a] rounded-full"></div>
                        <h2 className="text-sm font-bold text-[#1a1a1a] uppercase tracking-wider">Target Protein</h2>
                </div>
                      <div className="bg-white/40 rounded-lg p-4">
                        <h3 className="text-lg font-semibold text-[#1a1a1a] mb-2">{result.protein_info.name}</h3>
                        <div className="grid grid-cols-3 gap-4 text-xs mb-3">
                          <div>
                            <span className="text-[#888]">UniProt ID</span>
                            <p className="font-medium text-[#1a1a1a]">{result.protein_info.id}</p>
              </div>
                          <div>
                            <span className="text-[#888]">Organism</span>
                            <p className="font-medium text-[#1a1a1a]">{result.protein_info.organism}</p>
                  </div>
                      <div>
                            <span className="text-[#888]">Length</span>
                            <p className="font-medium text-[#1a1a1a]">{result.protein_info.sequence_length} aa</p>
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
                        <h2 className="text-sm font-bold text-[#1a1a1a] uppercase tracking-wider">Identified Compounds</h2>
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

                  {/* Analysis - with proper formatting and typing */}
                  {displayedText && (
                    <section className="mb-6">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="w-1 h-5 bg-[#1a1a1a] rounded-full"></div>
                        <h2 className="text-sm font-bold text-[#1a1a1a] uppercase tracking-wider">Research Analysis</h2>
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
                <Plus className="h-3.5 w-3.5" /> New
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
