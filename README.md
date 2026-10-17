# AppZar — Daniel Azar Portfolio & Engineering Blog

Unified personal website, interactive HTML5 canvas experiments, project portfolio, and Markdown engineering blog for **Daniel Azar** (`www.appzar.com.ar`), built with **Next.js 15 (Static Export)**, **React 19**, **TypeScript**, and **Tailwind CSS**.

## Features

- **Unified Single-Page Experience**: Combines the personal portfolio, interactive hero, blog, projects timeline, and bio into one cohesive page—eliminating the legacy external blog iframe.
- **Interactive Canvas Hero**:
  - [`GridCanvas.tsx`](./src/components/GridCanvas.tsx): Interactive neon grid background with mouse-reactive and ambient light trails.
  - [`ParticlePortrait.tsx`](./src/components/ParticlePortrait.tsx): Interactive particle portrait physics (`portrait.png`) that scatters on pointer hover/touch and springs back.
- **Native Markdown Blog**:
  - All posts live in [`content/posts/`](./content/posts) as Markdown (`.md`) files with YAML frontmatter.
  - Instant search, tag filtering, reading-time estimation, an inline single-page modal reader (`#post-<slug>`), and standalone static routes (`/posts/<slug>/`) for sharing & SEO.

## Adding a New Blog Post

Create a `.md` file in [`content/posts/`](./content/posts) with frontmatter:

```markdown
---
title: Your Post Title
subtitle: Optional subtitle
excerpt: Short summary displayed on the blog card.
date: '2026-10-16'
tags: ['Azure', 'TypeScript', 'Architecture']
thumb_img_path: /images/your-cover.png
content_img_path: /images/your-cover.png
---

Write your Markdown content here...
```

Place any post images inside [`public/images/`](./public/images).

## Development & Deployment

```bash
# Start local development server
npm run dev

# Type-check
npm run lint

# Build static export into ./out
npm run build

# Deploy ./out to gh-pages branch with CNAME www.appzar.com.ar
npm run deploy
```
