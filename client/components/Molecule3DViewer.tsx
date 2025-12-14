"use client";

import { useEffect, useRef, useState } from "react";

interface Molecule3DViewerProps {
  molBlock?: string;
  pdbContent?: string;
  width?: number;
  height?: number;
  style?: "stick" | "sphere" | "cartoon" | "line";
  backgroundColor?: string;
}

declare global {
  interface Window {
    $3Dmol: any;
  }
}

export default function Molecule3DViewer({
  molBlock,
  pdbContent,
  width = 300,
  height = 300,
  style = "stick",
  backgroundColor = "0xf5f0e8"
}: Molecule3DViewerProps) {
  const viewerRef = useRef<HTMLDivElement>(null);
  const [viewer, setViewer] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load 3Dmol.js from CDN
  useEffect(() => {
    if (typeof window !== "undefined" && !window.$3Dmol) {
      const script = document.createElement("script");
      script.src = "https://3dmol.org/build/3Dmol-min.js";
      script.async = true;
      script.onload = () => {
        setIsLoading(false);
      };
      script.onerror = () => {
        setError("Failed to load 3D viewer");
        setIsLoading(false);
      };
      document.head.appendChild(script);
    } else if (window.$3Dmol) {
      setIsLoading(false);
    }
  }, []);

  // Initialize viewer when 3Dmol.js is loaded
  useEffect(() => {
    if (isLoading || !viewerRef.current || !window.$3Dmol) return;

    try {
      // Create viewer
      const v = window.$3Dmol.createViewer(viewerRef.current, {
        backgroundColor: backgroundColor,
        antialias: true
      });
      setViewer(v);
      setError(null);
    } catch (e) {
      setError("Failed to initialize viewer");
    }
  }, [isLoading, backgroundColor]);

  // Load molecule when data changes
  useEffect(() => {
    if (!viewer) return;

    try {
      viewer.clear();

      if (molBlock) {
        viewer.addModel(molBlock, "mol");
      } else if (pdbContent) {
        viewer.addModel(pdbContent, "pdb");
      } else {
        return;
      }

      // Apply style
      switch (style) {
        case "sphere":
          viewer.setStyle({}, { sphere: { radius: 0.4 } });
          break;
        case "cartoon":
          viewer.setStyle({}, { cartoon: { color: "spectrum" } });
          break;
        case "line":
          viewer.setStyle({}, { line: { linewidth: 2 } });
          break;
        default:
          viewer.setStyle({}, { 
            stick: { radius: 0.15, colorscheme: "Jmol" },
            sphere: { radius: 0.3, colorscheme: "Jmol" }
          });
      }

      viewer.zoomTo();
      viewer.spin("y", 0.5);
      viewer.render();
      setError(null);
    } catch (e) {
      console.error("3D render error:", e);
      setError("Failed to render molecule");
    }
  }, [viewer, molBlock, pdbContent, style]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (viewer) {
        try {
          viewer.clear();
        } catch (e) {}
      }
    };
  }, [viewer]);

  if (error) {
    return (
      <div 
        style={{ width, height }} 
        className="flex items-center justify-center bg-[#ebe6dc] rounded-lg text-xs text-[#888]"
      >
        {error}
      </div>
    );
  }

  if (isLoading) {
    return (
      <div 
        style={{ width, height }} 
        className="flex items-center justify-center bg-[#ebe6dc] rounded-lg"
      >
        <div className="animate-spin h-6 w-6 border-2 border-[#888] border-t-transparent rounded-full"></div>
      </div>
    );
  }

  return (
    <div 
      ref={viewerRef} 
      style={{ width, height, position: "relative" }}
      className="rounded-lg overflow-hidden cursor-grab active:cursor-grabbing"
    />
  );
}

