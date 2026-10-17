# AppZar — Daniel Azar Portfolio & Engineering Blog

Unified personal portfolio, interactive HTML5 canvas experiments, project showcase, and Markdown engineering blog for **Daniel Azar** ([www.appzar.com.ar](https://www.appzar.com.ar)), built with **Next.js 15 (Static Export)**, **React 19**, **TypeScript**, and **Tailwind CSS**.

---

## Features

- **Unified Portfolio & Engineering Blog**: Combines the interactive Hero, 69+ deep-dive engineering articles, 22+ featured projects, and profile into a single fast static site.
- **Interactive Canvas Hero** (viewport-gated via `IntersectionObserver` so animations pause when scrolled off-screen):
  - [`GridCanvas.tsx`](./src/components/GridCanvas.tsx): Interactive neon grid background with mouse-reactive and ambient light trails.
  - [`ParticlePortrait.tsx`](./src/components/ParticlePortrait.tsx): Interactive particle portrait physics (`portrait.png`) that scatters on pointer hover/touch and springs back.
- **Zero-JS Server-Rendered Markdown & Math**:
  - All posts live in [`content/posts/`](./content/posts) as Markdown (`.md`) files with YAML frontmatter.
  - [`MarkdownRenderer.tsx`](./src/components/MarkdownRenderer.tsx) runs as a React Server Component at static build time, compiling GFM tables, KaTeX math (`$...$` / `$$...$$`), ASCII benchmark charts ([`DiagramBlockRenderer.tsx`](./src/components/DiagramBlockRenderer.tsx)), and interactive Mermaid diagrams ([`MermaidDiagram.tsx`](./src/components/MermaidDiagram.tsx)).
- **Automated WebP Cover, Thumbnail & Inline Asset Pipeline**:
  - [`scripts/optimize-images.mjs`](./scripts/optimize-images.mjs) generates `1280p` WebP covers (`public/images/covers/<slug>.webp`), `640×360` WebP card thumbnails (`public/images/thumbs/<slug>.webp`), dedicated project card thumbnails (`public/images/thumbs/project_*.webp`), and `.webp` siblings for inline images under `public/assets/**`.
- **Built-In SEO & Syndication Suite**:
  - Every article page ([`src/app/posts/[slug]/page.tsx`](./src/app/posts/[slug]/page.tsx)) emits `<link rel="canonical">`, OpenGraph `1280×720` cover metadata, Twitter `summary_large_image` cards, and `TechArticle` JSON-LD structured data.
  - [`scripts/generate-syndication.mjs`](./scripts/generate-syndication.mjs) generates [`public/rss.xml`](./public/rss.xml), [`public/sitemap.xml`](./public/sitemap.xml), [`public/robots.txt`](./public/robots.txt), and ready-to-publish **Dev.to** and **Medium** exports.

---

## Adding a New Blog Post

1. Create a `.md` file in [`content/posts/<slug>.md`](./content/posts) with YAML frontmatter:

```markdown
---
title: "Your Post Title"
subtitle: "Optional subtitle"
excerpt: "Short summary displayed on the blog card and OpenGraph description."
date: "2026-10-17"
tags: ["CUDA", "TensorRT", "AI"]
coverImage: "/images/covers/<slug>.webp"
---

Write your Markdown content here...
```

2. Place the cover image in `public/images/covers/<slug>.png` (or `.webp`) and any inline assets in `public/assets/<project>/`.
3. Run the image optimizer to generate the `640×360` WebP thumbnail and convert inline PNG/JPG assets to WebP:

```bash
npm run optimize-images
```

---

## Syndicating to Dev.to, Hashnode & Medium

Cross-posting articles to **Dev.to**, **Hashnode**, and **Medium** is fully supported and configured to preserve your **canonical URL** (`https://www.appzar.com.ar/posts/<slug>/`) so 100% of Google SEO authority stays on `www.appzar.com.ar`.

Running `npm run syndicate` (also executed automatically during `npm run build`) generates:
- **`public/rss.xml`**: Full-content RSS 2.0 + Atom feed (`<content:encoded>`) with all relative image paths (`/assets/...`) and internal post links (`/posts/...`) rewritten to absolute `https://www.appzar.com.ar/...` URLs.
- **`public/sitemap.xml`** & **`public/robots.txt`**: Complete XML sitemap of all canonical URLs.
- **`syndication/devto/<slug>.md`** *(git-ignored)*: Ready-to-publish Markdown files with Dev.to YAML frontmatter (`title`, `published: false`, `description`, max-4 alphanumeric `tags`, `cover_image`, and `canonical_url`).
- **`syndication/medium/<slug>.html`** *(git-ignored)*: Standalone HTML5 articles with `<link rel="canonical">`, OpenGraph tags, and LaTeX math symbols converted to clean Unicode (`∇f = 0`, `O(N³)`, `GF(2⁸)`) so Medium does not render raw `$...$` LaTeX source.

### 1. Connect Dev.to (Automatic RSS Sync or Direct API)

#### Option A: Automatic RSS Feed Import (Recommended)
1. Open **[https://dev.to/settings/extensions](https://dev.to/settings/extensions)** and scroll to **Publishing to DEV Community from RSS**.
2. Paste your feed URL:
   ```text
   https://www.appzar.com.ar/rss.xml
   ```
3. Check **"Mark the RSS source as canonical URL by default"** and click **Submit**.
4. Dev.to will automatically import your articles into **[https://dev.to/dashboard](https://dev.to/dashboard)** as private drafts with `canonical_url` and `cover_image` pre-filled. Review and click **Publish** on any draft you want to make public.

#### Option B: Direct CLI Publishing via Dev.to REST API
1. Generate a Dev.to API key at **[https://dev.to/settings/extensions](https://dev.to/settings/extensions)** (*DEV Community API Keys*).
2. Run the syndication script with `--publish-devto <slug>`:
   ```powershell
   # PowerShell
   $env:DEVTO_API_KEY="your_devto_api_key"
   node scripts/generate-syndication.mjs --publish-devto isurfer-part-2-equation-parser-ast-and-on-the-fly-glsl-compiler
   ```
   ```bash
   # Bash / macOS / Linux
   DEVTO_API_KEY="your_devto_api_key" node scripts/generate-syndication.mjs --publish-devto isurfer-part-2-equation-parser-ast-and-on-the-fly-glsl-compiler
   ```
   This posts the transformed Markdown from `syndication/devto/<slug>.md` to `https://dev.to/api/articles` as a draft with `canonical_url` and `main_image` set.

---

### 2. Connect Hashnode (Automatic RSS / GitHub Sync)

1. Open your **Hashnode Blog Dashboard** $\to$ **Import** $\to$ **RSS Importer**.
2. Paste:
   ```text
   https://www.appzar.com.ar/rss.xml
   ```
3. Hashnode will import each post with `originalArticleURL` (canonical URL) pointing to `https://www.appzar.com.ar/posts/<slug>/`, and will natively render both Markdown tables, KaTeX math (`$...$`), and Mermaid diagrams (` ```mermaid `).

---

### 3. Import Articles into Medium

Medium deprecated issuing new public API tokens, and its editor does not natively execute KaTeX math or Mermaid diagrams. You have two clean ways to publish to Medium while preserving your canonical URL:

1. **Direct URL Import (for standard prose/code articles)**:
   - Go to **[https://medium.com/p/import](https://medium.com/p/import)**.
   - Paste the live article URL, e.g.:
     ```text
     https://www.appzar.com.ar/posts/cinezero-autonomous-ai-filmmaking-studio/
     ```
   - Click **Import**. Medium automatically reads the `<link rel="canonical">`, `TechArticle` JSON-LD, and OpenGraph cover image from your page and sets the canonical link in Story Settings (*More settings $\to$ Advanced Settings $\to$ Customize Canonical Link*).
2. **Pre-Formatted Medium HTML (for math-heavy articles like `iSurfer` or `SurfZar`)**:
   - Run `npm run syndicate` and open `syndication/medium/<slug>.html` in your browser (where LaTeX expressions like `\nabla f = \mathbf{0}` and `O(N^3)` are already converted to clean Unicode `∇f = 0` and `O(N³)`).
   - Copy the rendered content into a new Medium story, and under **More settings $\to$ Advanced Settings $\to$ Customize Canonical Link**, paste `https://www.appzar.com.ar/posts/<slug>/`.

---

## Development & Deployment

```bash
# Start local development server
npm run dev

# Optimize cover images, 640x360 card thumbnails, project thumbnails & inline WebP assets
npm run optimize-images

# Generate public/rss.xml, public/sitemap.xml, and syndication/devto + syndication/medium exports
npm run syndicate

# Type-check TypeScript
npm run lint

# Build static export into ./out (automatically runs syndicate first)
npm run build

# Build and publish ./out to origin/gh-pages branch with CNAME www.appzar.com.ar
npm run deploy
```
