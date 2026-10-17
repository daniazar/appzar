'use client';

import React, { useState, useMemo } from 'react';
import Link from 'next/link';
import {
  BookOpen,
  Briefcase,
  User,
  Search,
  Calendar,
  Clock,
  ArrowRight,
  ExternalLink,
  Code2,
  Globe,
  Play,
  Video,
  Download,
  Sparkles,
  Mail,
  MapPin,
  Terminal,
  Layers,
  Cpu,
} from 'lucide-react';
import GridCanvas from './GridCanvas';
import ParticlePortrait from './ParticlePortrait';
import type { BlogPost } from '@/lib/posts';
import { PROFILE, PROJECTS, type ProjectItem } from '@/lib/data';

interface UnifiedHomeProps {
  posts: BlogPost[];
}

function formatDate(dateStr: string): string {
  const parsed = new Date(`${dateStr}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return dateStr;
  return parsed.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

function SocialSvgIcon({ type }: { type: string }) {
  switch (type) {
    case 'github':
      return (
        <svg viewBox="0 0 24 24" className="h-5 w-5 fill-current" aria-hidden="true">
          <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
        </svg>
      );
    case 'linkedin':
      return (
        <svg viewBox="0 0 24 24" className="h-5 w-5 fill-current" aria-hidden="true">
          <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
        </svg>
      );
    case 'twitter':
      return (
        <svg viewBox="0 0 24 24" className="h-5 w-5 fill-current" aria-hidden="true">
          <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
        </svg>
      );
    case 'instagram':
      return (
        <svg viewBox="0 0 24 24" className="h-5 w-5 fill-current" aria-hidden="true">
          <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z" />
        </svg>
      );
    case 'facebook':
      return (
        <svg viewBox="0 0 24 24" className="h-5 w-5 fill-current" aria-hidden="true">
          <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z" />
        </svg>
      );
    case 'whatsapp':
      return (
        <svg viewBox="0 0 24 24" className="h-5 w-5 fill-current" aria-hidden="true">
          <path d="M12.031 6.172c-3.181 0-5.767 2.586-5.768 5.766-.001 1.298.38 2.27 1.019 3.287l-.582 2.128 2.182-.573c.978.58 1.911.928 3.145.929 3.178 0 5.767-2.587 5.768-5.766.001-3.187-2.575-5.77-5.764-5.771zm3.392 8.244c-.144.405-.837.774-1.17.824-.299.045-.677.063-1.092-.069-.252-.08-.575-.187-.988-.365-1.739-.751-2.874-2.502-2.961-2.617-.087-.116-.708-.94-.708-1.793s.448-1.273.607-1.446c.159-.173.346-.217.462-.217l.332.006c.106.005.249-.04.39.298.144.347.491 1.2.534 1.287.043.087.072.188.014.304-.058.116-.087.188-.173.289l-.26.304c-.087.086-.177.18-.076.354.101.174.449.741.964 1.201.662.591 1.221.774 1.394.86s.274.072.376-.043c.101-.116.433-.506.549-.68.116-.173.231-.145.39-.087s1.011.477 1.184.564.289.13.332.202c.045.072.045.419-.1.824zm-3.423-14.416c-6.627 0-12 5.373-12 12 0 2.625.846 5.059 2.284 7.033l-1.494 5.462 5.605-1.469c1.911 1.246 4.19 1.974 6.605 1.974 6.627 0 12-5.373 12-12s-5.373-12-12-12z" />
        </svg>
      );
    default:
      return <ExternalLink className="h-5 w-5" />;
  }
}

function ProjectActionIcon({ icon }: { icon: ProjectItem['links'][number]['icon'] }) {
  switch (icon) {
    case 'code':
      return <Code2 className="h-4 w-4" />;
    case 'globe':
      return <Globe className="h-4 w-4" />;
    case 'video':
      return <Video className="h-4 w-4" />;
    case 'play':
      return <Play className="h-4 w-4" />;
    case 'download':
      return <Download className="h-4 w-4" />;
  }
}

const ACCENT_STYLES: Record<ProjectItem['accent'], { badge: string; border: string; glow: string }> = {
  cyan: {
    badge: 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30',
    border: 'hover:border-cyan-500/50',
    glow: 'from-cyan-500/20 via-transparent to-transparent',
  },
  purple: {
    badge: 'bg-purple-500/15 text-purple-300 border-purple-500/30',
    border: 'hover:border-purple-500/50',
    glow: 'from-purple-500/20 via-transparent to-transparent',
  },
  amber: {
    badge: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
    border: 'hover:border-amber-500/50',
    glow: 'from-amber-500/20 via-transparent to-transparent',
  },
  indigo: {
    badge: 'bg-indigo-500/15 text-indigo-300 border-indigo-500/30',
    border: 'hover:border-indigo-500/50',
    glow: 'from-indigo-500/20 via-transparent to-transparent',
  },
};

export default function UnifiedHome({ posts }: UnifiedHomeProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTag, setSelectedTag] = useState<string>('All');
  const [showAllPosts, setShowAllPosts] = useState(false);

  const allTags = useMemo(() => {
    const set = new Set<string>();
    posts.forEach((p) => p.tags.forEach((t) => set.add(t)));
    return ['All', ...Array.from(set)];
  }, [posts]);

  const filteredPosts = useMemo(() => {
    return posts.filter((post) => {
      const matchesTag = selectedTag === 'All' || post.tags.includes(selectedTag);
      const q = searchQuery.trim().toLowerCase();
      if (!q) return matchesTag;
      return (
        matchesTag &&
        (post.title.toLowerCase().includes(q) ||
          post.excerpt.toLowerCase().includes(q) ||
          post.content.toLowerCase().includes(q) ||
          post.tags.some((t) => t.toLowerCase().includes(q)))
      );
    });
  }, [posts, selectedTag, searchQuery]);

  const isFiltering = selectedTag !== 'All' || searchQuery.trim().length > 0;
  const visiblePosts = isFiltering || showAllPosts ? filteredPosts : filteredPosts.slice(0, 10);

  const handleEmailClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    const { name, domain, tld } = PROFILE.email;
    window.location.href = `mailto:${name}@${domain}.${tld}`;
  };

  return (
    <div className="min-h-screen bg-[#070a12] text-slate-100 selection:bg-[#4ea8de]/30 selection:text-[#80ffdb]">
      {/* Sticky Top Navbar */}
      <header className="sticky top-0 z-40 border-b border-slate-800/80 bg-[#070a12]/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
          <a href="#home" className="flex items-center gap-3 group">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src="/assets/Logo.png"
              alt="AppZar"
              className="h-9 w-auto object-contain transition-transform duration-300 group-hover:scale-105"
            />
            <div className="hidden sm:flex flex-col">
              <span className="text-sm font-bold tracking-tight text-white">Daniel Azar</span>
              <span className="text-[11px] text-slate-400">AppZar • Engineering &amp; Blog</span>
            </div>
          </a>

          <nav className="flex items-center gap-1 sm:gap-2">
            <a
              href="#blog"
              className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium text-slate-300 hover:bg-slate-800/70 hover:text-white transition-colors"
            >
              <BookOpen className="h-4 w-4 text-[#64dfdf]" />
              <span>Blog</span>
              <span className="ml-0.5 rounded-full bg-[#5e60ce]/25 border border-[#5e60ce]/40 px-1.5 py-0.2 text-[11px] font-semibold text-[#80ffdb]">
                {posts.length}
              </span>
            </a>
            <a
              href="#projects"
              className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium text-slate-300 hover:bg-slate-800/70 hover:text-white transition-colors"
            >
              <Briefcase className="h-4 w-4 text-[#4ea8de]" />
              <span>Projects</span>
            </a>
            <a
              href="#about"
              className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium text-slate-300 hover:bg-slate-800/70 hover:text-white transition-colors"
            >
              <User className="h-4 w-4 text-purple-400" />
              <span>About</span>
            </a>
          </nav>
        </div>
      </header>

      {/* Hero Section with Interactive Neon Grid + Particle Portrait */}
      <section
        id="home"
        className="relative overflow-hidden border-b border-slate-800/80 py-14 sm:py-20"
      >
        <GridCanvas />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-[#070a12]/30 via-transparent to-[#070a12]" />
        <div className="pointer-events-none absolute -top-28 left-1/2 -translate-x-1/2 h-96 w-[680px] rounded-full bg-gradient-to-r from-[#7400b8]/20 via-[#4ea8de]/15 to-[#80ffdb]/15 blur-3xl" />

        <div className="relative z-10 mx-auto max-w-4xl px-4 text-center sm:px-6">
          <ParticlePortrait />

          <div className="mx-auto mt-5 max-w-2xl rounded-2xl border border-slate-800/90 bg-[#070a12]/90 px-6 py-6 shadow-[0_0_50px_rgba(7,10,18,0.95)] backdrop-blur-md">
            <div className="inline-flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/95 px-3.5 py-1 text-xs font-medium text-slate-300">
              <Sparkles className="h-3.5 w-3.5 text-[#80ffdb]" />
              <span>{PROFILE.tagline}</span>
            </div>

            <h1 className="mt-3.5 text-4xl font-extrabold tracking-tight text-white sm:text-6xl">
              <span className="bg-gradient-to-r from-white via-slate-100 to-[#80ffdb] bg-clip-text text-transparent">
                {PROFILE.name}
              </span>
            </h1>

            <p className="mx-auto mt-3 max-w-xl text-base text-slate-200 sm:text-lg leading-relaxed">
              {PROFILE.role} with 10+ years building cloud-scale backends, interactive computer graphics,
              and real-time GPU neural rendering. Welcome to my unified portfolio &amp; engineering blog.
            </p>
          </div>

          {/* Social Links Bar */}
          <div className="mt-7 flex flex-wrap items-center justify-center gap-2.5">
            {PROFILE.socials.map((s) => (
              <a
                key={s.label}
                href={s.url}
                target="_blank"
                rel="noopener noreferrer"
                aria-label={s.label}
                title={s.label}
                className={`inline-flex items-center gap-2 rounded-xl border border-slate-800 bg-[#070a12]/95 px-3.5 py-2.5 text-slate-300 backdrop-blur transition-all duration-200 hover:-translate-y-0.5 ${s.hoverColor}`}
              >
                <SocialSvgIcon type={s.icon} />
                <span className="hidden sm:inline text-xs font-medium">{s.label}</span>
              </a>
            ))}
            <button
              type="button"
              onClick={handleEmailClick}
              title="Send Email"
              className="inline-flex items-center gap-2 rounded-xl border border-slate-800 bg-[#070a12]/95 px-3.5 py-2.5 text-slate-300 backdrop-blur transition-all duration-200 hover:-translate-y-0.5 hover:text-amber-400 hover:border-amber-400/50"
            >
              <Mail className="h-5 w-5" />
              <span className="hidden sm:inline text-xs font-medium">Email</span>
            </button>
          </div>

          {/* Quick CTAs */}
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <a
              href="#blog"
              className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-[#5e60ce] to-[#4ea8de] px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-[#5e60ce]/25 hover:brightness-110 transition-all"
            >
              <BookOpen className="h-4 w-4" />
              <span>Explore Blog Posts ({posts.length})</span>
            </a>
            <a
              href="#projects"
              className="inline-flex items-center gap-2 rounded-xl border border-slate-700/80 bg-slate-900/90 px-5 py-2.5 text-sm font-semibold text-slate-200 hover:border-slate-500 hover:bg-slate-800/80 transition-all"
            >
              <Briefcase className="h-4 w-4 text-[#80ffdb]" />
              <span>View Projects ({PROJECTS.length})</span>
            </a>
          </div>
        </div>
      </section>

      {/* Integrated Blog Section */}
      <section id="blog" className="scroll-mt-16 py-16 sm:py-20 border-b border-slate-800/80">
        <div className="mx-auto max-w-6xl px-4 sm:px-6">
          <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
            <div>
              <div className="inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-[#64dfdf]">
                <BookOpen className="h-4 w-4" />
                <span>Engineering Notes &amp; Deep-Dives</span>
              </div>
              <h2 className="mt-2 text-3xl font-bold tracking-tight text-white sm:text-4xl">
                Blog &amp; Articles
              </h2>
              <p className="mt-2 max-w-xl text-sm text-slate-400">
                Deep-dive engineering articles on CUDA Graphs, TensorRT FP8/FP16, neural talking-head avatars,
                WebGL2 algebraic surfaces, and cloud architecture.
              </p>
            </div>

            {/* Search Input */}
            <div className="relative w-full md:w-72">
              <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
              <input
                type="search"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search articles, topics..."
                className="w-full rounded-xl border border-slate-800 bg-slate-900/90 py-2.5 pl-10 pr-4 text-sm text-slate-100 placeholder-slate-500 focus:border-[#4ea8de] focus:outline-none focus:ring-1 focus:ring-[#4ea8de]"
              />
            </div>
          </div>

          {/* Tag Filters */}
          <div className="mt-6 flex flex-wrap items-center gap-2">
            {allTags.map((tag) => {
              const active = selectedTag === tag;
              return (
                <button
                  key={tag}
                  type="button"
                  onClick={() => setSelectedTag(tag)}
                  className={`rounded-full px-3.5 py-1.5 text-xs font-medium transition-all ${
                    active
                      ? 'bg-[#4ea8de] text-slate-950 font-semibold shadow-sm shadow-[#4ea8de]/30'
                      : 'border border-slate-800 bg-slate-900/70 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                  }`}
                >
                  {tag}
                </button>
              );
            })}
          </div>

          {/* Blog Posts Grid — Each card opens its dedicated page /posts/[slug] */}
          {filteredPosts.length === 0 ? (
            <div className="mt-10 rounded-2xl border border-slate-800/80 bg-slate-900/40 p-12 text-center">
              <p className="text-slate-400">No blog posts matched your filter.</p>
              <button
                type="button"
                onClick={() => {
                  setSearchQuery('');
                  setSelectedTag('All');
                }}
                className="mt-4 inline-flex items-center gap-2 rounded-lg bg-slate-800 px-4 py-2 text-xs font-medium text-[#80ffdb] hover:bg-slate-700"
              >
                Reset filters
              </button>
            </div>
          ) : (
            <>
              <div className="mt-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
                {visiblePosts.map((post, idx) => {
                  const isFeatured = idx === 0 && selectedTag === 'All' && !searchQuery;
                  return (
                    <Link
                      key={post.slug}
                      href={`/posts/${post.slug}`}
                      className={`group overflow-hidden rounded-2xl border border-slate-800/90 bg-slate-900/50 transition-all duration-300 hover:-translate-y-1 hover:border-[#4ea8de]/60 hover:bg-slate-900/90 hover:shadow-xl hover:shadow-[#4ea8de]/5 flex flex-col ${
                        isFeatured ? 'sm:col-span-2 lg:col-span-2' : ''
                      }`}
                    >
                      {post.thumbImg && (
                        <div
                          className={`relative overflow-hidden bg-slate-950 border-b border-slate-800/70 ${
                            isFeatured ? 'h-56 sm:h-64' : 'h-48'
                          }`}
                        >
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={post.thumbImg}
                            alt={post.title}
                            width={640}
                            height={360}
                            loading="lazy"
                            decoding="async"
                            className="h-full w-full object-cover opacity-90 transition-transform duration-500 group-hover:scale-105 group-hover:opacity-100"
                          />
                          <div className="absolute inset-0 bg-gradient-to-t from-slate-950/80 via-transparent to-transparent" />
                          <div className="absolute bottom-3 left-3 flex flex-wrap gap-1.5">
                            {post.tags.map((tag) => (
                              <span
                                key={tag}
                                className="rounded-md bg-slate-950/85 border border-slate-700/80 px-2.5 py-0.5 text-[11px] font-medium text-[#80ffdb] backdrop-blur-sm"
                              >
                                {tag}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      <div className="flex flex-1 flex-col justify-between p-6">
                        <div>
                          {!post.thumbImg && (
                            <div className="mb-3 flex flex-wrap gap-1.5">
                              {post.tags.map((tag) => (
                                <span
                                  key={tag}
                                  className="rounded-md bg-[#5e60ce]/15 border border-[#5e60ce]/30 px-2.5 py-0.5 text-[11px] font-medium text-[#80ffdb]"
                                >
                                  {tag}
                                </span>
                              ))}
                            </div>
                          )}

                          <div className="flex items-center gap-3 text-xs text-slate-400">
                            <span className="inline-flex items-center gap-1">
                              <Calendar className="h-3.5 w-3.5 text-[#4ea8de]" />
                              {formatDate(post.date)}
                            </span>
                            <span>•</span>
                            <span className="inline-flex items-center gap-1">
                              <Clock className="h-3.5 w-3.5 text-purple-400" />
                              {post.readingTime}
                            </span>
                          </div>

                          <h3
                            className={`mt-2.5 font-bold tracking-tight text-white group-hover:text-[#64dfdf] transition-colors ${
                              isFeatured ? 'text-xl sm:text-2xl' : 'text-lg'
                            }`}
                          >
                            {post.title}
                          </h3>

                          <p className="mt-2.5 line-clamp-3 text-sm leading-relaxed text-slate-400">
                            {post.excerpt}
                          </p>
                        </div>

                        <div className="mt-5 flex items-center justify-between pt-3 border-t border-slate-800/60">
                          <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-[#64dfdf] group-hover:translate-x-1 transition-transform">
                            <span>Read full article</span>
                            <ArrowRight className="h-3.5 w-3.5" />
                          </span>
                        </div>
                      </div>
                    </Link>
                  );
                })}
              </div>

              {!isFiltering && filteredPosts.length > 10 && (
                <div className="mt-10 flex justify-center">
                  <button
                    type="button"
                    onClick={() => setShowAllPosts((prev) => !prev)}
                    className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900/90 px-6 py-3 text-sm font-semibold text-white hover:border-[#4ea8de] hover:bg-slate-800 hover:text-[#80ffdb] transition-all"
                  >
                    <BookOpen className="h-4 w-4 text-[#64dfdf]" />
                    <span>
                      {showAllPosts
                        ? 'Show Featured Articles Only'
                        : `View All ${filteredPosts.length} Engineering Articles`}
                    </span>
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </section>

      {/* Projects Section */}
      <section id="projects" className="scroll-mt-16 py-16 sm:py-20 border-b border-slate-800/80">
        <div className="mx-auto max-w-6xl px-4 sm:px-6">
          <div>
            <div className="inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-[#4ea8de]">
              <Briefcase className="h-4 w-4" />
              <span>Portfolio &amp; Experiments</span>
            </div>
            <h2 className="mt-2 text-3xl font-bold tracking-tight text-white sm:text-4xl">
              Featured Projects
            </h2>
            <p className="mt-2 max-w-2xl text-sm text-slate-400">
              A timeline of applications, GPU shader renderers, games, and open-source tools I&apos;ve
              built over the years.
            </p>
          </div>

          <div className="mt-10 grid gap-6 md:grid-cols-2">
            {PROJECTS.map((project) => {
              const style = ACCENT_STYLES[project.accent];
              return (
                <div
                  key={project.id}
                  className={`group relative flex flex-col overflow-hidden rounded-2xl border border-slate-800/90 bg-slate-900/50 transition-all duration-300 ${style.border} hover:bg-slate-900/80`}
                >
                  <div
                    className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${style.glow} opacity-40`}
                  />

                  <div className="relative h-56 w-full overflow-hidden bg-[#05070d] border-b border-slate-800/80 flex items-center justify-center p-3">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={project.image}
                      alt={project.title}
                      loading="lazy"
                      className="max-h-full w-auto rounded-lg object-contain transition-transform duration-500 group-hover:scale-[1.03]"
                    />
                    <span
                      className={`absolute top-3 right-3 rounded-full border px-3 py-0.5 text-xs font-bold backdrop-blur-md ${style.badge}`}
                    >
                      {project.year}
                    </span>
                  </div>

                  <div className="relative flex flex-1 flex-col justify-between p-6">
                    <div>
                      <h3 className="text-xl font-bold text-white">{project.title}</h3>

                      <div className="mt-2.5 flex flex-wrap gap-1.5">
                        {project.tech.map((t) => (
                          <span
                            key={t}
                            className="rounded-md bg-slate-800/90 border border-slate-700/60 px-2 py-0.5 text-xs font-mono text-slate-300"
                          >
                            {t}
                          </span>
                        ))}
                      </div>

                      <p className="mt-3.5 text-sm leading-relaxed text-slate-300">
                        {project.description}
                      </p>

                      {project.note && (
                        <p className="mt-2.5 rounded-lg border border-amber-500/20 bg-amber-500/10 px-3 py-1.5 text-xs text-amber-200">
                          {project.note}
                        </p>
                      )}

                      {project.coAuthor && (
                        <p className="mt-2 text-xs font-medium text-slate-400">
                          {project.coAuthor}
                        </p>
                      )}
                    </div>

                    <div className="mt-6 flex flex-wrap gap-2.5 pt-4 border-t border-slate-800/70">
                      {project.links.map((link) => {
                        const isExternal = link.url.startsWith('http');
                        return (
                          <a
                            key={link.url}
                            href={link.url}
                            target={isExternal ? '_blank' : undefined}
                            rel={isExternal ? 'noopener noreferrer' : undefined}
                            className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-800/80 px-4 py-2 text-xs font-semibold text-white hover:border-[#4ea8de] hover:bg-slate-800 hover:text-[#80ffdb] transition-colors"
                          >
                            <ProjectActionIcon icon={link.icon} />
                            <span>{link.label}</span>
                          </a>
                        );
                      })}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* About & Skills Section */}
      <section id="about" className="scroll-mt-16 py-16 sm:py-20">
        <div className="mx-auto max-w-6xl px-4 sm:px-6">
          <div className="grid gap-10 lg:grid-cols-12">
            {/* Bio Column */}
            <div className="lg:col-span-7 space-y-5">
              <div className="inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-purple-400">
                <User className="h-4 w-4" />
                <span>About Me</span>
              </div>
              <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
                Engineering across AI, Computer Graphics &amp; Cloud
              </h2>

              <div className="inline-flex items-center gap-2 text-xs text-slate-400">
                <MapPin className="h-4 w-4 text-[#64dfdf]" />
                <span>{PROFILE.location}</span>
              </div>

              <div className="space-y-4 text-slate-300 leading-relaxed text-[15.5px]">
                {PROFILE.bio.map((paragraph, i) => (
                  <p key={i}>{paragraph}</p>
                ))}
              </div>
            </div>

            {/* Tech Stack Column */}
            <div className="lg:col-span-5 space-y-4">
              {PROFILE.skills.map((group, idx) => {
                const icons = [
                  <Terminal key="t" className="h-4 w-4 text-[#64dfdf]" />,
                  <Layers key="l" className="h-4 w-4 text-[#4ea8de]" />,
                  <Cpu key="c" className="h-4 w-4 text-purple-400" />,
                ];
                return (
                  <div
                    key={group.category}
                    className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-5"
                  >
                    <div className="flex items-center gap-2 text-sm font-bold text-white">
                      {icons[idx % icons.length]}
                      <span>{group.category}</span>
                    </div>
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {group.items.map((skill) => (
                        <span
                          key={skill}
                          className="rounded-lg border border-slate-800 bg-slate-950/80 px-2.5 py-1 text-xs font-medium text-slate-300"
                        >
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-[#05070d] py-10">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-4 sm:flex-row sm:px-6">
          <div className="flex items-center gap-3">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/assets/Logo.png" alt="AppZar" className="h-7 w-auto opacity-80" />
            <span className="text-xs text-slate-400">
              © {new Date().getFullYear()} {PROFILE.name} (AppZar). All rights reserved.
            </span>
          </div>
          <div className="flex items-center gap-4 text-xs text-slate-400">
            <a href="#home" className="hover:text-white transition-colors">
              Top
            </a>
            <a href="#blog" className="hover:text-white transition-colors">
              Blog
            </a>
            <a href="#projects" className="hover:text-white transition-colors">
              Projects
            </a>
            <a href="#about" className="hover:text-white transition-colors">
              About
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
