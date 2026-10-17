'use client';

import React, { useEffect, useId, useState } from 'react';
import { GitFork, Code } from 'lucide-react';
import DiagramBlockRenderer from './DiagramBlockRenderer';

interface MermaidDiagramProps {
  chart: string;
}

export default function MermaidDiagram({ chart }: MermaidDiagramProps) {
  const id = useId().replace(/[^a-zA-Z0-9]/g, '');
  const [svg, setSvg] = useState<string>('');
  const [error, setError] = useState<boolean>(false);
  const [showRaw, setShowRaw] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;
    async function renderChart() {
      try {
        const mermaid = (await import('mermaid')).default;
        mermaid.initialize({
          startOnLoad: false,
          theme: 'base',
          securityLevel: 'loose',
          fontFamily: 'Inter, system-ui, sans-serif',
          themeVariables: {
            darkMode: true,
            background: '#0b1020',
            primaryColor: '#131c33',
            primaryTextColor: '#f1f5f9',
            primaryBorderColor: '#4ea8de',
            lineColor: '#64dfdf',
            secondaryColor: '#1e1b4b',
            tertiaryColor: '#0f172a',
            noteBkgColor: '#1e293b',
            noteTextColor: '#e2e8f0',
            actorBkg: '#131c33',
            actorBorder: '#64dfdf',
            actorTextColor: '#f8fafc',
            signalColor: '#80ffdb',
            signalTextColor: '#f8fafc',
          },
        });
        const uniqueId = `mermaid-${id}-${Math.floor(Math.random() * 100000)}`;
        const { svg: renderedSvg } = await mermaid.render(uniqueId, chart.trim());
        if (!cancelled) {
          setSvg(renderedSvg);
          setError(false);
        }
      } catch {
        if (!cancelled) {
          setError(true);
        }
      }
    }
    renderChart();
    return () => {
      cancelled = true;
    };
  }, [chart, id]);

  if (error) {
    return <DiagramBlockRenderer rawText={chart} />;
  }

  return (
    <div className="my-6 overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-b from-[#0b1020] to-[#070a12] shadow-xl">
      <div className="flex items-center justify-between border-b border-slate-800/80 bg-slate-900/70 px-5 py-3">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-[#64dfdf]">
          <GitFork className="h-4 w-4" />
          <span>Interactive Architecture Diagram</span>
        </div>
        <button
          type="button"
          onClick={() => setShowRaw((prev) => !prev)}
          className="inline-flex items-center gap-1 rounded-md border border-slate-800 bg-slate-900 px-2.5 py-1 text-[11px] text-slate-400 hover:border-slate-700 hover:text-slate-200 transition-colors"
        >
          <Code className="h-3 w-3" />
          <span>{showRaw ? 'Diagram View' : 'Source'}</span>
        </button>
      </div>

      {showRaw ? (
        <pre className="overflow-x-auto p-5 font-mono text-xs leading-6 text-slate-300">
          <code>{chart}</code>
        </pre>
      ) : svg ? (
        <div
          className="flex justify-center overflow-x-auto p-6 [&_svg]:max-w-full [&_svg]:h-auto"
          dangerouslySetInnerHTML={{ __html: svg }}
        />
      ) : (
        <div className="flex h-40 items-center justify-center p-6 text-xs text-slate-400">
          Rendering architecture diagram...
        </div>
      )}
    </div>
  );
}
