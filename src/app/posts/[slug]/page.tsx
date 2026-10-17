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
} from 'lucide-react';
import { getAllPosts, getPostBySlug } from '@/lib/posts';
import MarkdownRenderer from '@/components/MarkdownRenderer';

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
    return { title: 'Post Not Found — Daniel Azar' };
  }
  return {
    title: `${post.title} — Daniel Azar`,
    description: post.excerpt,
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

  return (
    <div className="min-h-screen bg-[#070a12] text-slate-100 selection:bg-[#4ea8de]/30 selection:text-[#80ffdb]">
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
              <img src="/assets/Logo.png" alt="AppZar" className="h-7 w-auto" />
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

            <div className="mt-8 border-t border-slate-800/80 pt-8">
              <MarkdownRenderer content={post.content} />
            </div>

            {/* Next / Previous Article Cards */}
            <div className="mt-14 grid gap-4 border-t border-slate-800 pt-8 sm:grid-cols-2">
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
