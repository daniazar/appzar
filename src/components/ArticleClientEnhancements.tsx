'use client';

import React, { useState, useEffect } from 'react';
import { Check, Copy, Maximize2, X, Share2, Link2 } from 'lucide-react';

export function ReadingProgressBar() {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    let rafId = 0;
    const updateScroll = () => {
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        const scrollTop = window.scrollY;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        if (docHeight <= 0) {
          setProgress(0);
          return;
        }
        const pct = Math.min(100, Math.max(0, (scrollTop / docHeight) * 100));
        setProgress(pct);
      });
    };

    window.addEventListener('scroll', updateScroll, { passive: true });
    updateScroll();
    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener('scroll', updateScroll);
    };
  }, []);

  return (
    <div className="fixed top-0 left-0 right-0 z-50 h-1 bg-transparent pointer-events-none">
      <div
        className="h-full bg-gradient-to-r from-[#5e60ce] via-[#4ea8de] to-[#80ffdb] shadow-[0_0_10px_rgba(128,255,219,0.7)] transition-[width] duration-100 ease-out"
        style={{ width: `${progress}%` }}
      />
    </div>
  );
}

export function CopyCodeButton({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // Fallback if clipboard API fails
    }
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      title="Copy code to clipboard"
      className="inline-flex items-center gap-1 rounded-md border border-slate-700/80 bg-slate-900/90 px-2 py-0.5 text-[11px] font-sans font-medium text-slate-300 hover:border-[#4ea8de] hover:text-[#80ffdb] transition-colors"
    >
      {copied ? (
        <>
          <Check className="h-3 w-3 text-emerald-400" />
          <span className="text-emerald-400">Copied</span>
        </>
      ) : (
        <>
          <Copy className="h-3 w-3" />
          <span>Copy</span>
        </>
      )}
    </button>
  );
}

export function ZoomableImage({ src, alt }: { src?: string; alt?: string }) {
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsOpen(false);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen]);

  if (!src) return null;

  return (
    <>
      <span className="my-6 block overflow-hidden rounded-xl border border-slate-800 bg-slate-950/80 p-2">
        <span className="group relative block cursor-zoom-in" onClick={() => setIsOpen(true)}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={src}
            alt={alt || ''}
            loading="lazy"
            decoding="async"
            className="mx-auto max-h-[540px] w-auto rounded-lg object-contain transition-transform duration-300 group-hover:scale-[1.01]"
          />
          <span className="pointer-events-none absolute top-2.5 right-2.5 inline-flex items-center gap-1 rounded-lg border border-slate-700/80 bg-slate-950/85 px-2.5 py-1 text-[11px] font-medium text-slate-300 opacity-0 backdrop-blur transition-opacity group-hover:opacity-100">
            <Maximize2 className="h-3 w-3 text-[#80ffdb]" />
            <span>Click to zoom</span>
          </span>
        </span>
        {alt && <span className="mt-2 block text-center text-xs text-slate-400">{alt}</span>}
      </span>

      {isOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/95 p-4 backdrop-blur-md cursor-zoom-out"
          onClick={() => setIsOpen(false)}
        >
          <button
            type="button"
            onClick={() => setIsOpen(false)}
            aria-label="Close zoomed image"
            className="fixed top-5 right-5 rounded-xl border border-slate-700 bg-slate-900/90 p-2.5 text-slate-200 hover:border-[#4ea8de] hover:text-white transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
          <div className="max-h-[92vh] max-w-[94vw] overflow-auto">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={src}
              alt={alt || ''}
              className="mx-auto max-h-[86vh] w-auto rounded-xl border border-slate-800 object-contain shadow-2xl"
            />
            {alt && (
              <p className="mt-3 text-center text-xs sm:text-sm text-slate-300">{alt}</p>
            )}
          </div>
        </div>
      )}
    </>
  );
}

export function ArticleShareBar({ title, url }: { title: string; url: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopyUrl = async () => {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // ignore
    }
  };

  const linkedInShareUrl = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(url)}`;
  const twitterShareUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(`${title} by @azardani`)}&url=${encodeURIComponent(url)}`;

  return (
    <div className="mt-10 flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-800/90 bg-slate-900/40 px-5 py-4">
      <div className="flex items-center gap-2 text-xs sm:text-sm font-semibold text-slate-300">
        <Share2 className="h-4 w-4 text-[#64dfdf]" />
        <span>Share or cite this engineering article</span>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <a
          href={linkedInShareUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:border-[#0a66c2] hover:text-[#4ea8de] transition-colors"
        >
          <span>LinkedIn</span>
        </a>
        <a
          href={twitterShareUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:border-[#4099ff] hover:text-[#80ffdb] transition-colors"
        >
          <span>X / Twitter</span>
        </a>
        <button
          type="button"
          onClick={handleCopyUrl}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:border-[#4ea8de] hover:text-[#80ffdb] transition-colors"
        >
          {copied ? (
            <>
              <Check className="h-3.5 w-3.5 text-emerald-400" />
              <span className="text-emerald-400">Canonical Link Copied</span>
            </>
          ) : (
            <>
              <Link2 className="h-3.5 w-3.5 text-[#64dfdf]" />
              <span>Copy Link</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}
