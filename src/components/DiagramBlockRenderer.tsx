'use client';

import React, { useState } from 'react';
import { BarChart3, Cpu, GitBranch, Layers, Table2, Code } from 'lucide-react';

interface DiagramBlockRendererProps {
  rawText: string;
}

interface BarChartRow {
  label: string;
  barCount: number;
  valueText: string;
  highlight?: boolean;
}

interface ParsedBarChart {
  title: string;
  rows: BarChartRow[];
  maxBars: number;
}

interface ParsedAsciiTable {
  title?: string;
  headers: string[];
  rows: string[][];
  footer?: string[];
}

interface FlowStep {
  title: string;
  details: string[];
  parallel?: string[];
}

/**
 * 1. Detect & parse ASCII Horizontal Bar Charts containing '█'
 */
function tryParseBarChart(raw: string): ParsedBarChart | null {
  if (!raw.includes('█')) return null;
  const lines = raw
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);

  let title = 'Performance Benchmark';
  const rows: BarChartRow[] = [];

  for (const line of lines) {
    if (/^[+=-]+$/.test(line.replace(/\|/g, ''))) continue;
    const stripped = line.replace(/^\|\s*/, '').replace(/\s*\|$/, '').trim();
    if (!stripped) continue;

    if (stripped.includes('█')) {
      // e.g. "Librosa CPU Resample: ██████████ 1,041.5 ms"
      const match = stripped.match(/^([^:█]+):?\s*(█+)\s*(.*)$/);
      if (match) {
        const label = match[1].trim();
        const barCount = match[2].length;
        const valueText = match[3].trim();
        const highlight =
          /faster|cuda|tensorrt|trt|fp8|graph|opt|zero/i.test(label + ' ' + valueText) &&
          !/baseline|unoptimized|librosa/i.test(label);
        rows.push({ label, barCount, valueText, highlight });
      }
    } else if (rows.length === 0 && stripped.length > 3) {
      title = stripped;
    }
  }

  if (rows.length < 2) return null;
  const maxBars = Math.max(...rows.map((r) => r.barCount), 1);
  return { title, rows, maxBars };
}

/**
 * 2. Detect & parse ASCII Multi-Column Box Tables (+---+---+ / | Col1 | Col2 |)
 */
function tryParseAsciiTable(raw: string): ParsedAsciiTable | null {
  const lines = raw
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);

  // Must have multiple rows with at least 2 internal '|' separators
  const pipeRows = lines.filter((l) => l.startsWith('|') && l.endsWith('|'));
  if (pipeRows.length < 3) return null;

  const splitRow = (line: string): string[] =>
    line
      .slice(1, -1)
      .split('|')
      .map((c) => c.trim());

  let title: string | undefined;
  const multiColRows: string[][] = [];

  for (const line of lines) {
    if (/^[+=-]+$/.test(line.replace(/\|/g, ''))) continue;
    if (!line.startsWith('|') || !line.endsWith('|')) continue;
    const cols = splitRow(line);
    // Check if it's a separator line like |---|---|
    if (cols.every((c) => /^[-=]+$/.test(c))) continue;

    if (cols.length === 1 && multiColRows.length === 0) {
      title = cols[0];
    } else if (cols.length >= 2) {
      multiColRows.push(cols);
    }
  }

  if (multiColRows.length < 2) return null;
  const colCount = multiColRows[0].length;
  if (colCount < 2) return null;

  // Ensure consistent column count
  const validRows = multiColRows.filter((r) => r.length === colCount);
  if (validRows.length < 2) return null;

  const headers = validRows[0];
  const bodyRows = validRows.slice(1);
  let footer: string[] | undefined;

  if (
    bodyRows.length > 2 &&
    /^total/i.test(bodyRows[bodyRows.length - 1][0])
  ) {
    footer = bodyRows.pop();
  }

  return { title, headers, rows: bodyRows, footer };
}

/**
 * 3. Detect & parse ASCII Flowcharts / Architecture Boxes
 */
function isAsciiArchitectureDiagram(raw: string): boolean {
  return (
    raw.includes('+---') ||
    raw.includes('┌─') ||
    raw.includes('╔═') ||
    raw.includes('▼') ||
    raw.includes('──►') ||
    raw.includes('--->') ||
    raw.includes('====') ||
    (raw.includes('[') && raw.includes(']') && (raw.includes('│') || raw.includes('|') || raw.includes('v\n')))
  );
}

function parseArchitectureBox(raw: string): {
  title: string;
  sections: FlowStep[];
} {
  const rawLines = raw.split(/\r?\n/);
  let title = 'System Architecture & Pipeline Flow';
  const cleanedLines: string[] = [];

  for (const l of rawLines) {
    const trimmed = l.trim();
    if (!trimmed) continue;
    // Skip pure border lines
    if (/^[+=\-*┌┐└┘├┤┬┴┼─╔╗╚╝╠╣╦╩╬]+$/.test(trimmed)) continue;
    // Strip outer box pipes if present
    const inner = trimmed
      .replace(/^[|│║]\s*/, '')
      .replace(/\s*[|│║]$/, '')
      .trim();
    if (!inner) continue;
    if (/^[+=\-*─]+$/.test(inner)) continue;
    cleanedLines.push(inner);
  }

  if (
    cleanedLines.length > 0 &&
    cleanedLines[0].length < 95 &&
    !cleanedLines[0].startsWith('1.') &&
    !cleanedLines[0].startsWith('*') &&
    !cleanedLines[0].startsWith('-') &&
    (cleanedLines[0] === cleanedLines[0].toUpperCase() || cleanedLines[0].includes(':'))
  ) {
    title = cleanedLines.shift()!.replace(/:$/, '');
  }

  // Group into logical nodes/stages
  const sections: FlowStep[] = [];
  let current: FlowStep | null = null;

  for (const line of cleanedLines) {
    // Skip pure connector lines like "|", "v", "▼", "+---+---+"
    if (/^[|│▼v▲^┌┐└┘├┤┬┴┼─+\s/\\]+$/.test(line)) {
      continue;
    }

    // Check if line has parallel bracket nodes: e.g. "[Node A]   [Node B]"
    const bracketMatches = Array.from(line.matchAll(/\[\s*([^\]]+?)\s*\]/g)).map((m) => m[1]);
    if (bracketMatches.length >= 2 && line.replace(/\[[^\]]+\]/g, '').trim().length < 15) {
      sections.push({
        title: bracketMatches.join('  ⇄  '),
        details: [],
        parallel: bracketMatches,
      });
      current = null;
      continue;
    }

    // Check if line starts a numbered stage or a bracketed stage or uppercase header
    const numberedMatch = line.match(/^(\d+\.\s+[^:]+)(?::\s*(.*))?$/);
    const singleBracketMatch = line.match(/^\[\s*([^\]]+)\s*\]$/);

    if (numberedMatch) {
      current = {
        title: numberedMatch[1].trim(),
        details: numberedMatch[2] ? [numberedMatch[2].trim()] : [],
      };
      sections.push(current);
    } else if (singleBracketMatch) {
      current = {
        title: singleBracketMatch[1].trim(),
        details: [],
      };
      sections.push(current);
    } else if (/^(MODE [AB]|STAGE \d|THREAD \d|STEP \d)/i.test(line)) {
      current = {
        title: line,
        details: [],
      };
      sections.push(current);
    } else {
      if (!current) {
        current = {
          title: line.replace(/^[*•-]\s*/, ''),
          details: [],
        };
        sections.push(current);
      } else {
        current.details.push(line.replace(/^[*•]\s*/, ''));
      }
    }
  }

  return { title, sections };
}

export default function DiagramBlockRenderer({ rawText }: DiagramBlockRendererProps) {
  const [showRaw, setShowRaw] = useState(false);

  // 1. Try Bar Chart
  const barChart = tryParseBarChart(rawText);
  if (barChart && !showRaw) {
    return (
      <div className="my-6 overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-b from-[#0b1020] to-[#070a12] p-5 sm:p-6 shadow-xl">
        <div className="flex items-center justify-between gap-3 border-b border-slate-800/80 pb-3.5 mb-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#4ea8de]/15 border border-[#4ea8de]/30 text-[#64dfdf]">
              <BarChart3 className="h-4 w-4" />
            </div>
            <span className="text-xs sm:text-sm font-bold uppercase tracking-wider text-white">
              {barChart.title}
            </span>
          </div>
          <button
            type="button"
            onClick={() => setShowRaw(true)}
            className="text-[11px] text-slate-500 hover:text-slate-300 transition-colors"
          >
            Raw
          </button>
        </div>

        <div className="space-y-4">
          {barChart.rows.map((row, idx) => {
            const widthPct = Math.max(6, Math.round((row.barCount / barChart.maxBars) * 100));
            return (
              <div key={idx} className="space-y-1.5">
                <div className="flex flex-wrap items-baseline justify-between gap-2 text-xs sm:text-sm">
                  <span className="font-semibold text-slate-200">{row.label}</span>
                  <span
                    className={`font-mono text-xs font-bold px-2 py-0.5 rounded-md ${
                      row.highlight
                        ? 'bg-[#80ffdb]/15 text-[#80ffdb] border border-[#80ffdb]/30'
                        : 'bg-slate-800/90 text-slate-300'
                    }`}
                  >
                    {row.valueText}
                  </span>
                </div>
                <div className="h-3 w-full overflow-hidden rounded-full bg-slate-900 border border-slate-800/80 p-0.5">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      row.highlight
                        ? 'bg-gradient-to-r from-[#5e60ce] via-[#4ea8de] to-[#80ffdb] shadow-[0_0_12px_rgba(128,255,219,0.5)]'
                        : 'bg-gradient-to-r from-slate-600 to-slate-400'
                    }`}
                    style={{ width: `${widthPct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  // 2. Try ASCII Table
  const asciiTable = tryParseAsciiTable(rawText);
  if (asciiTable && !showRaw) {
    return (
      <div className="my-6 overflow-hidden rounded-2xl border border-slate-800 bg-[#0b1020] shadow-xl">
        {asciiTable.title && (
          <div className="flex items-center justify-between border-b border-slate-800 bg-slate-900/90 px-5 py-3.5">
            <div className="flex items-center gap-2.5">
              <Table2 className="h-4 w-4 text-[#64dfdf]" />
              <span className="text-xs sm:text-sm font-bold tracking-wider uppercase text-white">
                {asciiTable.title}
              </span>
            </div>
            <button
              type="button"
              onClick={() => setShowRaw(true)}
              className="text-[11px] text-slate-500 hover:text-slate-300"
            >
              Raw
            </button>
          </div>
        )}
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left text-xs sm:text-sm">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/60 text-[#80ffdb]">
                {asciiTable.headers.map((h, i) => (
                  <th key={i} className="px-4 py-3 font-semibold whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {asciiTable.rows.map((r, rIdx) => (
                <tr key={rIdx} className="hover:bg-slate-800/30 transition-colors">
                  {r.map((cell, cIdx) => (
                    <td
                      key={cIdx}
                      className={`px-4 py-2.5 ${
                        cIdx === 0 ? 'font-medium text-white' : 'font-mono text-slate-300'
                      }`}
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
            {asciiTable.footer && (
              <tfoot>
                <tr className="border-t-2 border-slate-700 bg-[#5e60ce]/15 font-bold text-white">
                  {asciiTable.footer.map((cell, i) => (
                    <td key={i} className="px-4 py-3 font-mono">
                      {cell}
                    </td>
                  ))}
                </tr>
              </tfoot>
            )}
          </table>
        </div>
      </div>
    );
  }

  // 3. Try Architecture / Flowchart Box
  if (isAsciiArchitectureDiagram(rawText) && !showRaw) {
    const { title, sections } = parseArchitectureBox(rawText);
    if (sections.length >= 2) {
      return (
        <div className="my-6 overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-b from-[#0b1020] to-[#070a12] p-5 sm:p-6 shadow-xl">
          <div className="flex items-center justify-between gap-3 border-b border-slate-800/80 pb-3.5 mb-5">
            <div className="flex items-center gap-2.5">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#5e60ce]/20 border border-[#5e60ce]/40 text-[#80ffdb]">
                <Layers className="h-4 w-4" />
              </div>
              <span className="text-xs sm:text-sm font-bold uppercase tracking-wider text-white">
                {title}
              </span>
            </div>
            <button
              type="button"
              onClick={() => setShowRaw(true)}
              className="inline-flex items-center gap-1 rounded-md border border-slate-800 bg-slate-900 px-2.5 py-1 text-[11px] text-slate-400 hover:border-slate-700 hover:text-slate-200 transition-colors"
            >
              <Code className="h-3 w-3" />
              <span>ASCII View</span>
            </button>
          </div>

          <div className="relative space-y-3">
            {sections.map((sec, idx) => (
              <React.Fragment key={idx}>
                {sec.parallel ? (
                  <div className="grid gap-3 sm:grid-cols-2">
                    {sec.parallel.map((pNode, pIdx) => (
                      <div
                        key={pIdx}
                        className="flex items-center gap-2.5 rounded-xl border border-[#4ea8de]/40 bg-slate-900/90 px-4 py-3 shadow-sm"
                      >
                        <GitBranch className="h-4 w-4 shrink-0 text-[#64dfdf]" />
                        <span className="font-mono text-xs sm:text-sm font-semibold text-white">
                          {pNode}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="rounded-xl border border-slate-800/90 bg-slate-900/75 p-4 hover:border-[#4ea8de]/40 transition-colors">
                    <div className="flex items-center gap-2.5">
                      <Cpu className="h-4 w-4 shrink-0 text-[#64dfdf]" />
                      <h4 className="font-mono text-xs sm:text-sm font-bold text-[#80ffdb]">
                        {sec.title}
                      </h4>
                    </div>
                    {sec.details.length > 0 && (
                      <div className="mt-2 space-y-1 pl-6 border-l-2 border-slate-800 ml-2">
                        {sec.details.map((d, dIdx) => (
                          <p key={dIdx} className="font-mono text-xs text-slate-300 leading-relaxed">
                            {d}
                          </p>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {idx < sections.length - 1 && (
                  <div className="flex justify-center py-0.5" aria-hidden="true">
                    <div className="h-4 w-0.5 bg-gradient-to-b from-[#4ea8de] to-[#7400b8]" />
                  </div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>
      );
    }
  }

  // Fallback or Raw toggle view
  return (
    <div className="relative my-5">
      {showRaw && (
        <button
          type="button"
          onClick={() => setShowRaw(false)}
          className="absolute right-3 top-3 z-10 rounded-md border border-slate-700 bg-slate-800/90 px-2.5 py-1 text-[11px] font-medium text-[#80ffdb] hover:bg-slate-700"
        >
          Visual View
        </button>
      )}
      <pre className="overflow-x-auto rounded-xl border border-slate-800 bg-[#070a12] p-4 shadow-inner font-mono text-[13px] leading-6 text-slate-100">
        <code>{rawText}</code>
      </pre>
    </div>
  );
}
