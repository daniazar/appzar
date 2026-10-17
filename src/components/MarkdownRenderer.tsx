import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeRaw from 'rehype-raw';
import rehypeKatex from 'rehype-katex';
import { normalizeAssetPath } from '@/lib/posts';
import MermaidDiagram from './MermaidDiagram';
import DiagramBlockRenderer from './DiagramBlockRenderer';

interface MarkdownRendererProps {
  content: string;
}

function normalizeImageUrl(src?: string): string | undefined {
  return normalizeAssetPath(src);
}

function normalizeHref(href?: string): string | undefined {
  if (!href) return undefined;
  const mdMatch = href.match(/(?:^|\/)([a-zA-Z0-9_-]+)\.md$/);
  if (mdMatch && !href.startsWith('http')) {
    return `/posts/${mdMatch[1]}`;
  }
  if ((href.startsWith('../') || href.startsWith('./')) && href.includes('assets/')) {
    return `/${href.replace(/^(\.\.\/|\.\/)+/, '')}`;
  }
  return href;
}

function looksLikeDiagramBlock(text: string, lang?: string): boolean {
  if (lang && !['text', 'plaintext', 'ascii'].includes(lang)) {
    return false;
  }
  return (
    text.includes('█') ||
    text.includes('+---') ||
    text.includes('┌─') ||
    text.includes('╔═') ||
    text.includes('▼') ||
    text.includes('──►') ||
    text.includes('--->') ||
    text.includes('====') ||
    (text.includes('[') && text.includes(']') && (text.includes('│') || text.includes('|')))
  );
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="markdown-body space-y-5 text-slate-300 leading-relaxed text-[16px]">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw, rehypeKatex]}
        components={{
          h1: ({ children }) => (
            <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight pt-6 pb-2 border-b border-slate-800/80">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-xl sm:text-2xl font-bold text-white tracking-tight pt-6 pb-1 flex items-center gap-2">
              <span className="inline-block h-5 w-1 shrink-0 rounded-full bg-gradient-to-b from-[#4ea8de] to-[#80ffdb]" />
              <span>{children}</span>
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-lg sm:text-xl font-semibold text-slate-100 pt-4">{children}</h3>
          ),
          h4: ({ children }) => (
            <h4 className="text-base sm:text-lg font-semibold text-slate-200 pt-3">{children}</h4>
          ),
          p: ({ children }) => <p className="text-slate-300 leading-7">{children}</p>,
          a: ({ href, children }) => {
            const resolvedHref = normalizeHref(href);
            const isExternal = resolvedHref?.startsWith('http');
            return (
              <a
                href={resolvedHref}
                target={isExternal ? '_blank' : undefined}
                rel={isExternal ? 'noopener noreferrer' : undefined}
                className="font-medium text-[#64dfdf] underline decoration-[#64dfdf]/40 underline-offset-4 hover:text-[#80ffdb] hover:decoration-[#80ffdb] transition-colors"
              >
                {children}
              </a>
            );
          },
          blockquote: ({ children }) => (
            <blockquote className="my-5 border-l-4 border-[#5e60ce] bg-slate-900/80 px-5 py-3.5 text-slate-200 rounded-r-xl space-y-1">
              {children}
            </blockquote>
          ),
          ul: ({ children }) => <ul className="list-disc pl-6 space-y-2 text-slate-300">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal pl-6 space-y-2 text-slate-300">{children}</ol>,
          li: ({ children }) => <li className="leading-7">{children}</li>,
          hr: () => <hr className="my-8 border-slate-800/80" />,
          pre: ({ children }) => <>{children}</>,
          code: ({ className, children, ...props }) => {
            const codeText = String(children).replace(/\n$/, '');
            const langMatch = className?.match(/language-([a-zA-Z0-9_-]+)/);
            const lang = langMatch ? langMatch[1].toLowerCase() : undefined;

            if (lang === 'mermaid') {
              return <MermaidDiagram chart={codeText} />;
            }

            const isBlock = Boolean(className) || String(children).includes('\n');
            if (!isBlock) {
              return (
                <code
                  className="rounded-md bg-slate-800/90 border border-slate-700/70 px-1.5 py-0.5 font-mono text-[13.5px] text-[#80ffdb]"
                  {...props}
                >
                  {children}
                </code>
              );
            }

            if (looksLikeDiagramBlock(codeText, lang)) {
              return <DiagramBlockRenderer rawText={codeText} />;
            }

            return (
              <pre className="my-5 overflow-x-auto rounded-xl border border-slate-800 bg-[#070a12] p-4 shadow-inner">
                {lang && (
                  <div className="mb-2 flex items-center justify-between border-b border-slate-800/80 pb-1.5 text-[11px] font-mono uppercase tracking-wider text-slate-500">
                    <span>{lang}</span>
                  </div>
                )}
                <code className="block font-mono text-[13px] leading-6 text-slate-100" {...props}>
                  {codeText}
                </code>
              </pre>
            );
          },
          table: ({ children }) => (
            <div className="my-6 overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full border-collapse text-left text-sm">{children}</table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-slate-900/90 text-slate-200 border-b border-slate-800">{children}</thead>
          ),
          th: ({ children }) => <th className="px-4 py-3 font-semibold">{children}</th>,
          td: ({ children }) => (
            <td className="border-t border-slate-800/70 px-4 py-2.5 text-slate-300">{children}</td>
          ),
          img: ({ src, alt }) => {
            const srcStr = typeof src === 'string' ? src : undefined;
            const normalizedSrc = normalizeImageUrl(srcStr);
            return (
              <span className="my-6 block overflow-hidden rounded-xl border border-slate-800/90 bg-slate-950/60 p-2 shadow-lg">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={normalizedSrc}
                  alt={alt || ''}
                  loading="lazy"
                  decoding="async"
                  className="mx-auto max-h-[560px] w-auto rounded-lg object-contain"
                />
                {alt && (
                  <span className="mt-2 block text-center text-xs text-slate-400 italic">{alt}</span>
                )}
              </span>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
