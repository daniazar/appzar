import React from 'react';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import {
  ArrowLeft,
  BookOpen,
  Briefcase,
  User,
  Calendar,
  Clock,
  ChevronLeft,
  ChevronRight,
  List,
  Layers,
} from 'lucide-react';
import { getAllPosts, getPostBySlug, type BlogPost } from '@/lib/posts';
import { slugifyHeading } from '@/lib/syntaxHighlight';
import MarkdownRenderer from '@/components/MarkdownRenderer';
import { ReadingProgressBar, ArticleShareBar } from '@/components/ArticleClientEnhancements';

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || 'https://www.appzar.com.ar';

interface TocHeading {
  level: 2 | 3;
  text: string;
  id: string;
}

function extractHeadings(markdown: string): TocHeading[] {
  const headings: TocHeading[] = [];
  let inCodeBlock = false;
  for (const rawLine of markdown.split(/\r?\n/)) {
    const trimmed = rawLine.trim();
    if (trimmed.startsWith('```')) {
      inCodeBlock = !inCodeBlock;
      continue;
    }
    if (inCodeBlock) continue;

    const match = trimmed.match(/^(#{2,3})\s+(.+)$/);
    if (match) {
      const level = match[1].length as 2 | 3;
      const cleanText = match[2]
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
        .replace(/[`*_~]/g, '')
        .trim();
      const id = slugifyHeading(cleanText);
      if (id && cleanText) {
        headings.push({ level, text: cleanText, id });
      }
    }
  }
  return headings;
}

const SERIES_DEFINITIONS: Array<{
  name: string;
  match: (slug: string) => boolean;
}> = [
  {
    name: 'iSurfer: Real-Time GPU Algebraic Surface Ray Tracer (ITBA Thesis Series)',
    match: (s) => s.startsWith('isurfer-'),
  },
  {
    name: 'Molten AI Agent: Autonomous Business Analyst & Fibonacci Engine (MBA Thesis Series)',
    match: (s) => s.startsWith('molten-ai-'),
  },
  {
    name: 'SurfZar: Real-Time WebGL2 Algebraic Surfaces, CSG & 3D Printing Series',
    match: (s) => s.startsWith('surfzar'),
  },
  {
    name: 'ArchitecturaAI: Browser-Native 3D Spatial Studio & NODES Image Tree Series',
    match: (s) => s.startsWith('architectura-ai-'),
  },
  {
    name: 'Ruby NewsStudio: MuseTalk Real-Time TensorRT & CUDA Avatar Series',
    match: (s) => s.startsWith('engineering_blog_musetalk_'),
  },
  {
    name: 'Ruby NewsStudio: Ditto Motion-Space TensorRT & Blackwell FP8 Series',
    match: (s) => s.startsWith('engineering_blog_ditto_') || /^1[6-9]_/.test(s) || /^2[0-1]_/.test(s),
  },
  {
    name: 'Ruby NewsStudio: Kokoro 82M Sub-100ms Neural TTS Series',
    match: (s) => s.startsWith('engineering_blog_kokoro_'),
  },
  {
    name: 'Ruby NewsStudio: Chatterbox Zero-Shot Voice Cloning & CUDA Graphs Series',
    match: (s) => s.startsWith('engineering_blog_chatterbox_'),
  },
  {
    name: 'Ruby NewsStudio: Qwen3-TTS Dual-Codebook & TensorRT Series',
    match: (s) => s.startsWith('engineering_blog_qwen3_'),
  },
  {
    name: 'Ruby NewsStudio: Fish Audio S1-Mini & DAC Vocoder Optimization Series',
    match: (s) => s.startsWith('engineering_blog_fish_'),
  },
];

export function generateStaticParams() {
  const posts = getAllPosts();
  return posts.map((post) => ({
    slug: post.slug,
  }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const post = getPostBySlug(slug);
  if (!post) {
    return { title: 'Post Not Found' };
  }
  const canonicalUrl = `${SITE_URL}/posts/${post.slug}/`;
  const ogImage = post.contentImg || post.thumbImg || '/images/thumbs/project_ruby_newsstudio.webp';
  const absoluteOgImage = ogImage.startsWith('http') ? ogImage : `${SITE_URL}${ogImage}`;

  return {
    title: post.title,
    description: post.excerpt,
    authors: [{ name: 'Daniel Azar', url: SITE_URL }],
    keywords: post.tags,
    alternates: {
      canonical: canonicalUrl,
    },
    openGraph: {
      type: 'article',
      url: canonicalUrl,
      title: post.title,
      description: post.excerpt,
      siteName: 'Daniel Azar — AppZar Engineering Blog',
      publishedTime: `${post.date}T00:00:00.000Z`,
      authors: ['Daniel Azar'],
      tags: post.tags,
      images: [
        {
          url: absoluteOgImage,
          width: 1280,
          height: 720,
          alt: post.title,
        },
      ],
    },
    twitter: {
      card: 'summary_large_image',
      creator: '@azardani',
      title: post.title,
      description: post.excerpt,
      images: [absoluteOgImage],
    },
  };
}

function formatDate(dateStr: string): string {
  const parsed = new Date(`${dateStr}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return dateStr;
  return parsed.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
}

export default async function BlogPostPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const posts = getAllPosts();
  const postIndex = posts.findIndex((p) => p.slug === slug);
  if (postIndex === -1) {
    notFound();
  }

  const post = posts[postIndex];
  const prevPost = postIndex > 0 ? posts[postIndex - 1] : null;
  const nextPost = postIndex < posts.length - 1 ? posts[postIndex + 1] : null;

  const activeSeriesDef = SERIES_DEFINITIONS.find((def) => def.match(post.slug));
  const seriesPosts: BlogPost[] = activeSeriesDef
    ? [...posts].filter((p) => activeSeriesDef.match(p.slug)).reverse()
    : [];
  const currentSeriesPartIndex = seriesPosts.findIndex((p) => p.slug === post.slug);

  const headings = extractHeadings(post.content);

  const seriesPrefixes = [
    'isurfer-',
    'molten-ai-',
    'surfzar',
    'architectura-ai-',
    'engineering_blog_musetalk_',
    'engineering_blog_ditto_',
    'engineering_blog_kokoro_',
    'engineering_blog_chatterbox_',
    'engineering_blog_qwen3_',
    'engineering_blog_fish_',
  ];
  const matchedPrefix = seriesPrefixes.find((prefix) => post.slug.startsWith(prefix));
  const relatedPosts = posts
    .filter((p) => p.slug !== post.slug)
    .map((candidate) => {
      let score = 0;
      if (matchedPrefix && candidate.slug.startsWith(matchedPrefix)) score += 10;
      const sharedTags = candidate.tags.filter((t) => post.tags.includes(t)).length;
      score += sharedTags * 2;
      return { candidate, score };
    })
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3)
    .map((item) => item.candidate);

  const canonicalUrl = `${SITE_URL}/posts/${post.slug}/`;
  const heroImage = post.contentImg || post.thumbImg;
  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline: post.title,
    description: post.excerpt,
    datePublished: `${post.date}T00:00:00.000Z`,
    dateModified: `${post.date}T00:00:00.000Z`,
    url: canonicalUrl,
    mainEntityOfPage: {
      '@type': 'WebPage',
      '@id': canonicalUrl,
    },
    author: {
      '@type': 'Person',
      name: 'Daniel Azar',
      url: SITE_URL,
    },
    publisher: {
      '@type': 'Person',
      name: 'Daniel Azar',
      url: SITE_URL,
    },
    image: heroImage ? [`${SITE_URL}${heroImage}`] : undefined,
    keywords: post.tags.join(', '),
  };

  return (
    <div className="min-h-screen bg-[#070a12] text-slate-100 selection:bg-[#4ea8de]/30 selection:text-[#80ffdb]">
      <ReadingProgressBar />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      {/* Full Site Sticky Navbar */}
      <header className="sticky top-0 z-40 border-b border-slate-800/80 bg-[#070a12]/85 backdrop-blur-xl">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <Link
              href="/#blog"
              className="inline-flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900/80 px-3.5 py-2 text-xs font-semibold text-slate-200 hover:border-[#4ea8de] hover:text-[#80ffdb] transition-colors"
            >
              <ArrowLeft className="h-4 w-4" />
              <span>All Articles</span>
            </Link>
            <Link href="/" className="hidden sm:flex items-center gap-2.5 pl-2">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="/assets/Logo.webp"
                alt="AppZar"
                width={28}
                height={28}
                decoding="async"
                className="h-7 w-auto"
              />
              <span className="text-sm font-bold text-white">Daniel Azar</span>
            </Link>
          </div>

          <nav className="flex items-center gap-1 sm:gap-2">
            <Link
              href="/#blog"
              className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs sm:text-sm font-medium text-slate-300 hover:bg-slate-800/70 hover:text-white transition-colors"
            >
              <BookOpen className="h-4 w-4 text-[#64dfdf]" />
              <span>Blog</span>
            </Link>
            <Link
              href="/#projects"
              className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs sm:text-sm font-medium text-slate-300 hover:bg-slate-800/70 hover:text-white transition-colors"
            >
              <Briefcase className="h-4 w-4 text-[#4ea8de]" />
              <span>Projects</span>
            </Link>
            <Link
              href="/#about"
              className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs sm:text-sm font-medium text-slate-300 hover:bg-slate-800/70 hover:text-white transition-colors"
            >
              <User className="h-4 w-4 text-purple-400" />
              <span>About</span>
            </Link>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-4xl px-4 py-10 sm:px-6 sm:py-14">
        <article className="overflow-hidden rounded-2xl border border-slate-800/90 bg-[#0b0f1c] shadow-2xl">
          {post.contentImg && (
            <div className="relative max-h-[420px] w-full overflow-hidden border-b border-slate-800 bg-slate-950 flex items-center justify-center">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={post.contentImg}
                alt={post.title}
                decoding="async"
                fetchPriority="high"
                className="max-h-[420px] w-full object-cover"
              />
              <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-[#0b0f1c] via-transparent to-transparent opacity-80" />
            </div>
          )}

          <div className="p-6 sm:p-12">
            {/* Tags & Metadata */}
            <div className="flex flex-wrap items-center gap-2 mb-5">
              {post.tags.map((tag) => (
                <span
                  key={tag}
                  className="rounded-md bg-[#5e60ce]/20 border border-[#5e60ce]/40 px-2.5 py-0.5 text-xs font-semibold text-[#80ffdb]"
                >
                  {tag}
                </span>
              ))}
              <span className="text-xs text-slate-400 ml-1 inline-flex items-center gap-1">
                <Calendar className="h-3.5 w-3.5 text-[#4ea8de]" />
                {formatDate(post.date)}
              </span>
              <span className="text-xs text-slate-400 inline-flex items-center gap-1">
                • <Clock className="h-3.5 w-3.5 text-purple-400" />
                {post.readingTime}
              </span>
            </div>

            <h1 className="text-2xl sm:text-4xl md:text-5xl font-extrabold tracking-tight text-white leading-tight">
              {post.title}
            </h1>

            {post.subtitle && (
              <p className="mt-3 text-base sm:text-lg text-slate-400 leading-relaxed">
                {post.subtitle}
              </p>
            )}

            {/* Multi-Part Series Banner */}
            {activeSeriesDef && seriesPosts.length > 1 && (
              <div className="mt-6 rounded-xl border border-[#4ea8de]/30 bg-slate-900/70 p-4 sm:p-5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2 text-xs sm:text-sm font-bold text-[#80ffdb]">
                    <Layers className="h-4 w-4 text-[#4ea8de] shrink-0" />
                    <span>{activeSeriesDef.name}</span>
                  </div>
                  <span className="rounded-full border border-[#4ea8de]/40 bg-[#4ea8de]/15 px-2.5 py-0.5 text-[11px] font-semibold text-white">
                    Part {currentSeriesPartIndex + 1} of {seriesPosts.length}
                  </span>
                </div>
                <div className="mt-3 grid gap-1.5 sm:grid-cols-2">
                  {seriesPosts.map((sp, idx) => {
                    const isCurrent = sp.slug === post.slug;
                    return (
                      <Link
                        key={sp.slug}
                        href={`/posts/${sp.slug}`}
                        className={`flex items-center gap-2 rounded-lg px-3 py-2 text-xs transition-colors ${
                          isCurrent
                            ? 'border border-[#4ea8de]/50 bg-[#4ea8de]/15 font-bold text-white'
                            : 'border border-slate-800/80 bg-slate-950/60 text-slate-300 hover:border-slate-700 hover:text-[#80ffdb]'
                        }`}
                      >
                        <span className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-slate-800 text-[11px] font-mono text-[#64dfdf]">
                          {idx + 1}
                        </span>
                        <span className="line-clamp-1">{sp.title}</span>
                      </Link>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Collapsible Table of Contents (if >= 3 headings) */}
            {headings.length >= 3 && (
              <details
                open={headings.length <= 8}
                className="group mt-6 rounded-xl border border-slate-800/90 bg-slate-900/50 p-4 sm:p-5"
              >
                <summary className="flex cursor-pointer list-none items-center justify-between text-xs sm:text-sm font-bold uppercase tracking-wider text-slate-200">
                  <span className="inline-flex items-center gap-2">
                    <List className="h-4 w-4 text-[#64dfdf]" />
                    <span>Table of Contents ({headings.length} sections)</span>
                  </span>
                  <span className="text-xs font-medium text-slate-400 group-open:rotate-90 transition-transform">
                    ▸
                  </span>
                </summary>
                <nav className="mt-3 border-t border-slate-800/80 pt-3">
                  <ul className="space-y-1.5 text-xs sm:text-sm">
                    {headings.map((h) => (
                      <li
                        key={h.id}
                        className={h.level === 3 ? 'pl-4 text-slate-400' : 'font-medium text-slate-300'}
                      >
                        <a
                          href={`#${h.id}`}
                          className="inline-flex items-center gap-1.5 hover:text-[#80ffdb] transition-colors"
                        >
                          <span className="text-[#4ea8de]/70">•</span>
                          <span>{h.text}</span>
                        </a>
                      </li>
                    ))}
                  </ul>
                </nav>
              </details>
            )}

            <div className="mt-8 border-t border-slate-800/80 pt-8">
              <MarkdownRenderer content={post.content} />
            </div>

            {/* Social Share & Copy Canonical URL Bar */}
            <ArticleShareBar title={post.title} url={canonicalUrl} />

            {/* Related / Companion Articles in Series or Topic */}
            {relatedPosts.length > 0 && (
              <div className="mt-14 border-t border-slate-800 pt-8">
                <h2 className="text-sm font-bold uppercase tracking-wider text-[#80ffdb]">
                  Related Articles &amp; Companion Deep-Dives
                </h2>
                <div className="mt-4 grid gap-4 sm:grid-cols-3">
                  {relatedPosts.map((rel) => (
                    <Link
                      key={rel.slug}
                      href={`/posts/${rel.slug}`}
                      className="group flex flex-col justify-between rounded-xl border border-slate-800/90 bg-slate-900/50 p-4 hover:border-[#4ea8de]/60 hover:bg-slate-900 transition-all"
                    >
                      <div>
                        <span className="text-[11px] font-medium text-slate-400">
                          {formatDate(rel.date)} • {rel.readingTime}
                        </span>
                        <h3 className="mt-1.5 text-sm font-bold text-white group-hover:text-[#64dfdf] transition-colors line-clamp-3">
                          {rel.title}
                        </h3>
                      </div>
                      <span className="mt-3 inline-flex items-center gap-1 text-xs font-semibold text-[#64dfdf]">
                        <span>Read article</span>
                        <ChevronRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
                      </span>
                    </Link>
                  ))}
                </div>
              </div>
            )}

            {/* Next / Previous Article Cards */}
            <div className="mt-10 grid gap-4 border-t border-slate-800 pt-8 sm:grid-cols-2">
              {prevPost ? (
                <Link
                  href={`/posts/${prevPost.slug}`}
                  className="group flex flex-col justify-between rounded-xl border border-slate-800 bg-slate-900/60 p-5 text-left hover:border-[#4ea8de]/60 hover:bg-slate-900 transition-all"
                >
                  <span className="inline-flex items-center gap-1 text-xs font-semibold text-[#64dfdf]">
                    <ChevronLeft className="h-3.5 w-3.5 transition-transform group-hover:-translate-x-0.5" />
                    Newer Article
                  </span>
                  <span className="mt-2 font-bold text-sm sm:text-base text-white group-hover:text-[#80ffdb] transition-colors line-clamp-2">
                    {prevPost.title}
                  </span>
                </Link>
              ) : (
                <div />
              )}
              {nextPost && (
                <Link
                  href={`/posts/${nextPost.slug}`}
                  className="group flex flex-col justify-between items-end rounded-xl border border-slate-800 bg-slate-900/60 p-5 text-right hover:border-[#4ea8de]/60 hover:bg-slate-900 transition-all"
                >
                  <span className="inline-flex items-center gap-1 text-xs font-semibold text-[#64dfdf]">
                    Older Article
                    <ChevronRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
                  </span>
                  <span className="mt-2 font-bold text-sm sm:text-base text-white group-hover:text-[#80ffdb] transition-colors line-clamp-2">
                    {nextPost.title}
                  </span>
                </Link>
              )}
            </div>
          </div>
        </article>
      </main>
    </div>
  );
}
