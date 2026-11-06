import React from 'react';
import type { Metadata } from 'next';
import Link from 'next/link';
import {
  ArrowLeft,
  Globe,
  MapPin,
  Mail,
  Briefcase,
  GraduationCap,
  Award,
  Code2,
  BookOpen,
  ExternalLink,
} from 'lucide-react';
import { PROFILE, PROJECTS, FLAGSHIP_PAPERS } from '@/lib/data';
import PrintResumeButton from '@/components/PrintResumeButton';

export const metadata: Metadata = {
  title: 'Resume & CV — Daniel Azar (Software AI Engineer)',
  description:
    'Curriculum Vitae of Daniel Azar — Software AI Engineer (Scale AI, MoltenMinds, SOUTHWORKS / Microsoft Azure, Globant, MBA 9.59/10, ITBA Software Engineering).',
  alternates: {
    canonical: 'https://www.appzar.com.ar/resume/',
  },
};

export default function ResumePage() {
  const emailAddress = `${PROFILE.email.name}@${PROFILE.email.domain}.${PROFILE.email.tld}`;
  const topProjects = PROJECTS.slice(0, 6);

  return (
    <div className="min-h-screen bg-[#070a12] text-slate-100 print:bg-white print:text-slate-900">
      {/* Top Action Bar (Hidden when printing to PDF) */}
      <header className="sticky top-0 z-40 border-b border-slate-800/80 bg-[#070a12]/90 backdrop-blur-xl print:hidden">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="inline-flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900/80 px-3.5 py-2 text-xs font-semibold text-slate-200 hover:border-[#4ea8de] hover:text-[#80ffdb] transition-colors"
            >
              <ArrowLeft className="h-4 w-4" />
              <span>Back to Portfolio</span>
            </Link>
            <span className="hidden sm:inline text-xs text-slate-400">
              ATS-Ready &amp; Print-Optimized Resume
            </span>
          </div>

          <div className="flex items-center gap-2.5">
            <a
              href="https://ar.linkedin.com/in/daniazar/"
              target="_blank"
              rel="noopener noreferrer"
              className="hidden sm:inline-flex items-center gap-1.5 rounded-xl border border-slate-800 bg-slate-900 px-3.5 py-2 text-xs font-semibold text-slate-200 hover:border-[#0a66c2] hover:text-white transition-colors"
            >
              <span>LinkedIn</span>
              <ExternalLink className="h-3.5 w-3.5" />
            </a>
            <PrintResumeButton />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-4xl px-4 py-10 sm:px-6 print:max-w-none print:px-0 print:py-0">
        <div className="rounded-2xl border border-slate-800/90 bg-[#0b0f1c] p-6 sm:p-12 shadow-2xl print:rounded-none print:border-none print:bg-white print:p-2 print:shadow-none">
          {/* Header / Contact Info */}
          <div className="border-b border-slate-800 pb-6 print:border-slate-300">
            <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
              <div>
                <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white print:text-slate-950">
                  {PROFILE.name}
                </h1>
                <p className="mt-1 text-base sm:text-lg font-semibold text-[#64dfdf] print:text-slate-800">
                  {PROFILE.role} • Generative AI, 3D Computer Graphics &amp; Cloud Systems
                </p>
                <p className="mt-1 inline-flex items-center gap-1.5 text-xs text-slate-400 print:text-slate-600">
                  <MapPin className="h-3.5 w-3.5 text-[#4ea8de] print:text-slate-700" />
                  <span>{PROFILE.location}</span>
                </p>
              </div>

              <div className="flex flex-col gap-1 text-xs text-slate-300 sm:text-right print:text-slate-700">
                <a
                  href={`mailto:${emailAddress}`}
                  className="inline-flex items-center gap-1.5 sm:justify-end hover:text-[#80ffdb] print:text-slate-900"
                >
                  <Mail className="h-3.5 w-3.5 text-[#64dfdf] print:hidden" />
                  <span>{emailAddress}</span>
                </a>
                <a
                  href="https://www.appzar.com.ar"
                  className="inline-flex items-center gap-1.5 sm:justify-end hover:text-[#80ffdb] print:text-slate-900"
                >
                  <Globe className="h-3.5 w-3.5 text-[#4ea8de] print:hidden" />
                  <span>www.appzar.com.ar</span>
                </a>
                <a
                  href="https://github.com/daniazar"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-[#80ffdb] print:text-slate-900"
                >
                  github.com/daniazar
                </a>
                <a
                  href="https://ar.linkedin.com/in/daniazar/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-[#80ffdb] print:text-slate-900"
                >
                  linkedin.com/in/daniazar
                </a>
              </div>
            </div>

            {/* Executive Summary */}
            <p className="mt-4 text-sm leading-relaxed text-slate-300 print:text-slate-800">
              {PROFILE.heroSummary} Holds an <strong>MBA (GPA 9.59/10.0)</strong> from Universidad de Palermo and a{' '}
              <strong>Software Engineering degree (Computer Graphics specialization)</strong> from ITBA. Author of 69+
              deep-dive engineering articles on Blackwell TensorRT FP8/FP16, CUDA Graphs, WebGL2 ray tracing, and
              autonomous multi-agent systems.
            </p>
          </div>

          {/* Work Experience */}
          <section className="mt-7">
            <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-[#80ffdb] print:text-slate-950">
              <Briefcase className="h-4 w-4 text-[#4ea8de] print:text-slate-800" />
              <span>Professional Experience</span>
            </h2>

            <div className="mt-4 space-y-5">
              {PROFILE.timeline.map((exp) => (
                <div key={exp.period} className="break-inside-avoid">
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <div>
                      <h3 className="text-base font-bold text-white print:text-slate-950">{exp.role}</h3>
                      <p className="text-xs font-semibold text-[#64dfdf] print:text-slate-700">{exp.org}</p>
                    </div>
                    <span className="text-xs font-semibold text-slate-400 print:text-slate-600">{exp.period}</span>
                  </div>
                  {exp.bullets ? (
                    <ul className="mt-2 list-disc space-y-1 pl-5 text-xs sm:text-sm text-slate-300 leading-relaxed print:text-slate-800">
                      {exp.bullets.map((b, idx) => (
                        <li key={idx}>{b}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-1.5 text-xs sm:text-sm text-slate-300 leading-relaxed print:text-slate-800">
                      {exp.description}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </section>

          {/* Education */}
          <section className="mt-8 border-t border-slate-800/80 pt-6 print:border-slate-300">
            <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-[#80ffdb] print:text-slate-950">
              <GraduationCap className="h-4 w-4 text-[#4ea8de] print:text-slate-800" />
              <span>Education &amp; Academic Theses</span>
            </h2>

            <div className="mt-4 space-y-4">
              {PROFILE.education.map((edu) => (
                <div key={edu.degree} className="break-inside-avoid">
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <h3 className="text-sm sm:text-base font-bold text-white print:text-slate-950">{edu.degree}</h3>
                    <span className="text-xs font-semibold text-slate-400 print:text-slate-600">{edu.year}</span>
                  </div>
                  <p className="text-xs font-medium text-slate-400 print:text-slate-700">{edu.institution}</p>
                  <p className="mt-0.5 text-xs text-[#64dfdf] print:text-slate-800">{edu.thesis}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Selected Flagship Systems & Research */}
          <section className="mt-8 border-t border-slate-800/80 pt-6 print:border-slate-300">
            <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-[#80ffdb] print:text-slate-950">
              <Code2 className="h-4 w-4 text-[#4ea8de] print:text-slate-800" />
              <span>Selected AI, GPU &amp; 3D Systems</span>
            </h2>

            <div className="mt-4 grid gap-3 sm:grid-cols-2 print:grid-cols-2">
              {topProjects.map((proj) => (
                <div
                  key={proj.id}
                  className="break-inside-avoid rounded-xl border border-slate-800/90 bg-slate-900/50 p-3.5 print:border-slate-300 print:bg-white"
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <h3 className="text-xs sm:text-sm font-bold text-white print:text-slate-950">{proj.title}</h3>
                    <span className="text-[11px] font-semibold text-[#64dfdf] print:text-slate-700">{proj.year}</span>
                  </div>
                  <p className="mt-1 text-xs text-slate-300 leading-relaxed print:text-slate-800">
                    {proj.description}
                  </p>
                  <p className="mt-1.5 text-[11px] font-mono text-slate-400 print:text-slate-600">
                    {proj.tech.join(' • ')}
                  </p>
                </div>
              ))}
            </div>
          </section>

          {/* Flagship Engineering Publications */}
          <section className="mt-8 border-t border-slate-800/80 pt-6 print:border-slate-300">
            <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-[#80ffdb] print:text-slate-950">
              <BookOpen className="h-4 w-4 text-[#4ea8de] print:text-slate-800" />
              <span>Selected Engineering Publications (69 Articles on appzar.com.ar)</span>
            </h2>
            <ul className="mt-3 space-y-2 text-xs sm:text-sm">
              {FLAGSHIP_PAPERS.map((paper) => (
                <li key={paper.url} className="break-inside-avoid">
                  <Link
                    href={paper.url}
                    className="font-semibold text-white hover:text-[#80ffdb] print:text-slate-950"
                  >
                    [{paper.metric}] {paper.title}
                  </Link>
                  <span className="block text-xs text-slate-400 print:text-slate-700">{paper.summary}</span>
                </li>
              ))}
            </ul>
          </section>

          {/* Honors, Skills & Languages */}
          <section className="mt-8 grid gap-6 border-t border-slate-800/80 pt-6 sm:grid-cols-2 print:grid-cols-2 print:border-slate-300">
            <div>
              <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-[#80ffdb] print:text-slate-950">
                <Award className="h-4 w-4 text-amber-400 print:text-slate-800" />
                <span>Honors &amp; Awards</span>
              </h2>
              <ul className="mt-3 space-y-2 text-xs text-slate-300 print:text-slate-800">
                {PROFILE.awards.map((a) => (
                  <li key={a.title} className="break-inside-avoid">
                    <strong className="text-white print:text-slate-950">
                      {a.title} ({a.year}):
                    </strong>{' '}
                    {a.detail}
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <h2 className="text-sm font-bold uppercase tracking-wider text-[#80ffdb] print:text-slate-950">
                Core Technical Stack &amp; Languages
              </h2>
              <div className="mt-3 space-y-2 text-xs text-slate-300 print:text-slate-800">
                {PROFILE.skills.map((s) => (
                  <p key={s.category}>
                    <strong className="text-white print:text-slate-950">{s.category}:</strong>{' '}
                    {s.items.join(', ')}
                  </p>
                ))}
                <p>
                  <strong className="text-white print:text-slate-950">Languages:</strong>{' '}
                  {PROFILE.languages.map((l) => `${l.name} (${l.level})`).join(' • ')}
                </p>
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
