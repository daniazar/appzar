import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

const SITE_URL = (process.env.NEXT_PUBLIC_SITE_URL || 'https://www.appzar.com.ar').replace(/\/+$/, '');
const rootDir = process.cwd();
const postsDir = path.join(rootDir, 'content', 'posts');
const publicDir = path.join(rootDir, 'public');
const syndicationDir = path.join(rootDir, 'syndication');
const devtoOutDir = path.join(syndicationDir, 'devto');
const mediumOutDir = path.join(syndicationDir, 'medium');

function escapeXml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function resolveAbsoluteAssetUrl(rawPath) {
  if (!rawPath) return null;
  const cleaned = rawPath.trim().replace(/^(\.\.\/|\.\/)+/, '');
  if (cleaned.startsWith('http://') || cleaned.startsWith('https://')) {
    return cleaned;
  }
  const rel = cleaned.startsWith('/') ? cleaned : `/${cleaned}`;
  if (/\.(png|jpg|jpeg)$/i.test(rel)) {
    const webpCandidate = rel.replace(/\.(png|jpg|jpeg)$/i, '.webp');
    const diskWebp = path.join(publicDir, webpCandidate.replace(/^\//, ''));
    if (fs.existsSync(diskWebp)) {
      return `${SITE_URL}${webpCandidate}`;
    }
  }
  return `${SITE_URL}${rel}`;
}

function extractTitleAndStripH1(rawContent, frontmatterTitle) {
  const lines = rawContent.split(/\r?\n/);
  let title = frontmatterTitle || '';
  let startIdx = 0;
  while (startIdx < lines.length && lines[startIdx].trim() === '') startIdx++;
  if (startIdx < lines.length && lines[startIdx].startsWith('# ')) {
    const h1Text = lines[startIdx].replace(/^#\s+/, '').trim();
    if (!title) title = h1Text;
    lines.splice(startIdx, 1);
  }
  return { title: title || 'Untitled Post', body: lines.join('\n').trim() };
}

function extractCleanExcerpt(markdown, fallbackTitle) {
  const lines = markdown.split(/\r?\n/);
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (
      !line ||
      line.startsWith('#') ||
      line.startsWith('---') ||
      line.startsWith('![') ||
      line.startsWith('```') ||
      line.startsWith('|') ||
      line.startsWith('>') ||
      line.startsWith('- ') ||
      line.startsWith('* ') ||
      /^\d+\.\s/.test(line) ||
      /^\*\*(Author|Date|Hardware|Category|Target|ComfyUI|Next\.js|Artifact|Deliverable|Companion|Stack|Environment|Evaluation|Topic)/i.test(
        line
      )
    ) {
      continue;
    }
    const plain = line
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/\*\*([^*]+)\*\*/g, '$1')
      .replace(/\*([^*]+)\*/g, '$1')
      .replace(/`([^`]+)`/g, '$1')
      .replace(/\$[^$]+\$/g, '')
      .replace(/\s+/g, ' ')
      .trim();
    if (plain.length >= 45) {
      return plain.length > 200 ? `${plain.slice(0, 197)}...` : plain;
    }
  }
  return fallbackTitle;
}

function resolvePostDate(slug, markdown, frontmatterDate) {
  if (frontmatterDate) return { date: String(frontmatterDate), seriesOrder: 0 };
  const numMatch = slug.match(/^(\d{2})_/);
  if (numMatch) {
    const partNum = parseInt(numMatch[1], 10);
    const baseDate = new Date(Date.UTC(2026, 7, 28 + partNum));
    const yyyy = baseDate.getUTCFullYear();
    const mm = String(baseDate.getUTCMonth() + 1).padStart(2, '0');
    const dd = String(baseDate.getUTCDate()).padStart(2, '0');
    return { date: `${yyyy}-${mm}-${dd}`, seriesOrder: 100 + partNum };
  }
  if (slug.startsWith('engineering_blog_')) {
    if (slug.includes('webgpu') || slug.includes('piper')) {
      return { date: '2026-10-05', seriesOrder: 200 };
    }
    const partMatch = slug.match(/part_(\d+)/);
    const p = partMatch ? parseInt(partMatch[1], 10) : 1;
    return { date: `2026-09-${String(10 + p).padStart(2, '0')}`, seriesOrder: 50 + p };
  }
  const dateLineMatch = markdown.match(/\*\*Date\*\*:\s*([^\r\n]+)/i);
  if (dateLineMatch) {
    const parsed = new Date(dateLineMatch[1].replace(/\*+/g, '').trim());
    if (!Number.isNaN(parsed.getTime())) {
      return { date: parsed.toISOString().slice(0, 10), seriesOrder: 10 };
    }
  }
  return { date: '2026-09-15', seriesOrder: 1 };
}

/**
 * Rewrite relative Markdown images and internal links to absolute canonical URLs
 */
function rewriteMarkdownLinksToAbsolute(markdown) {
  // 1. Rewrite Markdown images ![alt](path)
  let out = markdown.replace(/!\[([^\]]*)\]\(([^)\s]+)(\s+"[^"]*")?\)/g, (_m, alt, url, titlePart) => {
    const absUrl = resolveAbsoluteAssetUrl(url);
    return `![${alt}](${absUrl}${titlePart || ''})`;
  });

  // 2. Rewrite Markdown links [label](href)
  out = out.replace(/(?<!!)\[([^\]]+)\]\(([^)\s]+)\)/g, (m, label, href) => {
    if (href.startsWith('http://') || href.startsWith('https://') || href.startsWith('#') || href.startsWith('mailto:')) {
      return m;
    }
    const mdMatch = href.match(/(?:^|\/)([a-zA-Z0-9_-]+)\.md$/);
    if (mdMatch) {
      return `[${label}](${SITE_URL}/posts/${mdMatch[1]}/)`;
    }
    if (href.startsWith('/posts/')) {
      const cleanPost = href.replace(/\/+$/, '');
      return `[${label}](${SITE_URL}${cleanPost}/)`;
    }
    if ((href.startsWith('../') || href.startsWith('./') || href.startsWith('/')) && href.includes('assets/')) {
      return `[${label}](${resolveAbsoluteAssetUrl(href)})`;
    }
    if (href.startsWith('/')) {
      return `[${label}](${SITE_URL}${href})`;
    }
    return m;
  });

  return out;
}

/**
 * Map post metadata to Dev.to's strict max-4 lowercase alphanumeric tags
 */
function inferDevtoTags(slug, title, body) {
  const combined = `${slug} ${title} ${body.slice(0, 1000)}`.toLowerCase();
  const tags = [];
  if (combined.includes('cuda') || combined.includes('tensorrt') || combined.includes('triton')) tags.push('cuda');
  if (/\bai\b/.test(combined) || combined.includes('neural') || combined.includes('agent') || combined.includes('llm'))
    tags.push('ai');
  if (combined.includes('webgl') || combined.includes('opengl') || combined.includes('glsl') || combined.includes('ray'))
    tags.push('computergraphics');
  if (combined.includes('typescript') || combined.includes('next.js') || combined.includes('react'))
    tags.push('typescript');
  if (combined.includes('webgpu')) tags.push('webgpu');
  if (combined.includes('python') || combined.includes('pytorch')) tags.push('python');
  if (combined.includes('cloud') || combined.includes('azure') || combined.includes('security')) tags.push('cloud');
  if (combined.includes('gamedev') || combined.includes('unity') || combined.includes('aoe2')) tags.push('gamedev');
  if (slug.includes('isurfer')) tags.push('ios', 'shaders');
  if (tags.length === 0) tags.push('programming', 'webdev');
  return Array.from(new Set(tags)).slice(0, 4);
}

/**
 * Convert inline/block LaTeX to clean Unicode math for Medium (which lacks native LaTeX rendering)
 */
function latexToUnicodeForMedium(text) {
  return text
    .replace(/\\nabla/g, '∇')
    .replace(/\\partial/g, '∂')
    .replace(/\\mathbf\{([^}]+)\}/g, '$1')
    .replace(/\\text\{([^}]+)\}/g, '$1')
    .replace(/\\frac\{([^}]+)\}\{([^}]+)\}/g, '($1)/($2)')
    .replace(/\\sqrt\{([^}]+)\}/g, '√($1)')
    .replace(/\\cdot/g, '·')
    .replace(/\\times/g, '×')
    .replace(/\\to/g, '→')
    .replace(/\\le/g, '≤')
    .replace(/\\ge/g, '≥')
    .replace(/\\neq/g, '≠')
    .replace(/\\approx/g, '≈')
    .replace(/\\infty/g, '∞')
    .replace(/\\alpha/g, 'α')
    .replace(/\\beta/g, 'β')
    .replace(/\\gamma/g, 'γ')
    .replace(/\\delta/g, 'δ')
    .replace(/\\epsilon/g, 'ε')
    .replace(/\\lambda/g, 'λ')
    .replace(/\\phi/g, 'φ')
    .replace(/\\tau/g, 'τ')
    .replace(/\\theta/g, 'θ')
    .replace(/\^2\b/g, '²')
    .replace(/\^3\b/g, '³')
    .replace(/\^4\b/g, '⁴')
    .replace(/\^6\b/g, '⁶')
    .replace(/\^8\b/g, '⁸');
}

function simpleMarkdownToMediumHtml(md, post) {
  const cleanedMd = latexToUnicodeForMedium(md);
  const lines = cleanedMd.split(/\r?\n/);
  const html = [];
  let inCode = false;
  let codeLines = [];

  for (const line of lines) {
    if (line.trim().startsWith('```')) {
      if (!inCode) {
        inCode = true;
        codeLines = [];
      } else {
        inCode = false;
        html.push(`<pre><code>${escapeXml(codeLines.join('\n'))}</code></pre>`);
      }
      continue;
    }
    if (inCode) {
      codeLines.push(line);
      continue;
    }
    const t = line.trim();
    if (!t) continue;
    if (t.startsWith('### ')) {
      html.push(`<h3>${escapeXml(t.slice(4))}</h3>`);
    } else if (t.startsWith('## ')) {
      html.push(`<h2>${escapeXml(t.slice(3))}</h2>`);
    } else if (t.startsWith('# ')) {
      html.push(`<h1>${escapeXml(t.slice(2))}</h1>`);
    } else if (t.startsWith('![')) {
      const m = t.match(/!\[([^\]]*)\]\(([^)\s]+)\)/);
      if (m) {
        html.push(
          `<figure><img src="${escapeXml(m[2])}" alt="${escapeXml(m[1])}" /><figcaption>${escapeXml(
            m[1]
          )}</figcaption></figure>`
        );
      }
    } else if (t.startsWith('> ')) {
      html.push(`<blockquote>${escapeXml(t.slice(2))}</blockquote>`);
    } else {
      const formatted = escapeXml(t)
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
      html.push(`<p>${formatted}</p>`);
    }
  }

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>${escapeXml(post.title)}</title>
  <meta name="description" content="${escapeXml(post.excerpt)}" />
  <meta name="author" content="Daniel Azar" />
  <link rel="canonical" href="${post.canonicalUrl}" />
  <meta property="og:title" content="${escapeXml(post.title)}" />
  <meta property="og:description" content="${escapeXml(post.excerpt)}" />
  <meta property="og:url" content="${post.canonicalUrl}" />
  <meta property="og:image" content="${post.coverUrl}" />
</head>
<body>
  <article>
    <h1>${escapeXml(post.title)}</h1>
    ${post.subtitle ? `<p><em>${escapeXml(post.subtitle)}</em></p>` : ''}
    <figure><img src="${post.coverUrl}" alt="${escapeXml(post.title)}" /></figure>
    ${html.join('\n    ')}
    <hr />
    <p><em>Originally published at <a href="${post.canonicalUrl}">${post.canonicalUrl}</a>.</em></p>
  </article>
</body>
</html>`;
}

async function publishToDevto(post, devtoMarkdown, apiKey) {
  const res = await fetch('https://dev.to/api/articles', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'api-key': apiKey,
    },
    body: JSON.stringify({
      article: {
        title: post.title,
        body_markdown: devtoMarkdown,
        published: false,
        main_image: post.coverUrl,
        canonical_url: post.canonicalUrl,
        description: post.excerpt,
        tags: post.devtoTags,
      },
    }),
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Dev.to API error (${res.status}): ${errText}`);
  }
  return await res.json();
}

async function main() {
  fs.mkdirSync(devtoOutDir, { recursive: true });
  fs.mkdirSync(mediumOutDir, { recursive: true });

  const files = fs.readdirSync(postsDir).filter((f) => f.endsWith('.md'));
  const posts = files.map((file) => {
    const slug = file.replace(/\.md$/, '');
    const raw = fs.readFileSync(path.join(postsDir, file), 'utf8');
    const { data, content: rawBody } = matter(raw);
    const { title, body } = extractTitleAndStripH1(rawBody, data.title);
    const { date, seriesOrder } = resolvePostDate(slug, rawBody, data.date);
    const excerpt = data.excerpt || data.subtitle || extractCleanExcerpt(body, title);

    const dedicatedCoverWebp = path.join(publicDir, 'images', 'covers', `${slug}.webp`);
    const dedicatedThumbWebp = path.join(publicDir, 'images', 'thumbs', `${slug}.webp`);
    const coverRel = fs.existsSync(dedicatedCoverWebp)
      ? `/images/covers/${slug}.webp`
      : fs.existsSync(dedicatedThumbWebp)
        ? `/images/thumbs/${slug}.webp`
        : data.content_img_path || data.thumb_img_path || '/images/thumbs/project_ruby_newsstudio.webp';

    const coverUrl = resolveAbsoluteAssetUrl(coverRel);
    const canonicalUrl = `${SITE_URL}/posts/${slug}/`;
    const absoluteBody = rewriteMarkdownLinksToAbsolute(body);
    const devtoTags = inferDevtoTags(slug, title, body);

    return {
      slug,
      title,
      subtitle: data.subtitle,
      excerpt,
      date,
      seriesOrder,
      coverUrl,
      canonicalUrl,
      absoluteBody,
      devtoTags,
    };
  });

  posts.sort((a, b) => {
    if (a.date !== b.date) return a.date < b.date ? 1 : -1;
    return (b.seriesOrder || 0) - (a.seriesOrder || 0);
  });

  // 1. Write Dev.to Markdown & Medium HTML files for every post
  for (const post of posts) {
    const devtoContent = `---
title: ${JSON.stringify(post.title)}
published: false
description: ${JSON.stringify(post.excerpt.slice(0, 150))}
tags: ${post.devtoTags.join(', ')}
cover_image: ${post.coverUrl}
canonical_url: ${post.canonicalUrl}
---

${post.absoluteBody}

---

*Originally published at [appzar.com.ar](${post.canonicalUrl}).*
`;
    fs.writeFileSync(path.join(devtoOutDir, `${post.slug}.md`), devtoContent, 'utf8');

    const mediumHtml = simpleMarkdownToMediumHtml(post.absoluteBody, post);
    fs.writeFileSync(path.join(mediumOutDir, `${post.slug}.html`), mediumHtml, 'utf8');
  }

  // 2. Generate public/rss.xml (Full-content RSS 2.0 + Atom + Media for Dev.to & Hashnode auto-import)
  const rssItems = posts
    .map((post) => {
      const pubDate = new Date(`${post.date}T12:00:00Z`).toUTCString();
      return `    <item>
      <title>${escapeXml(post.title)}</title>
      <link>${post.canonicalUrl}</link>
      <guid isPermaLink="true">${post.canonicalUrl}</guid>
      <pubDate>${pubDate}</pubDate>
      <dc:creator><![CDATA[Daniel Azar]]></dc:creator>
      <description>${escapeXml(post.excerpt)}</description>
      <media:thumbnail url="${escapeXml(post.coverUrl)}" />
      <enclosure url="${escapeXml(post.coverUrl)}" type="image/webp" length="0" />
      <content:encoded><![CDATA[${post.absoluteBody}]]></content:encoded>
    </item>`;
    })
    .join('\n');

  const rssXml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
  xmlns:dc="http://purl.org/dc/elements/1.1/"
  xmlns:content="http://purl.org/rss/1.0/modules/content/"
  xmlns:atom="http://www.w3.org/2005/Atom"
  xmlns:media="http://search.yahoo.com/mrss/">
  <channel>
    <title>Daniel Azar — AppZar Engineering Blog</title>
    <description>Deep-dive engineering articles on CUDA Graphs, TensorRT FP8/FP16, neural talking-head avatars, WebGL2 algebraic surfaces, and cloud architecture.</description>
    <link>${SITE_URL}/</link>
    <atom:link href="${SITE_URL}/rss.xml" rel="self" type="application/rss+xml" />
    <language>en-us</language>
    <lastBuildDate>${new Date().toUTCString()}</lastBuildDate>
${rssItems}
  </channel>
</rss>
`;
  fs.writeFileSync(path.join(publicDir, 'rss.xml'), rssXml, 'utf8');

  // 3. Generate public/sitemap.xml & public/robots.txt for Google canonical indexing
  const sitemapUrls = [
    `  <url>
    <loc>${SITE_URL}/</loc>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>`,
    ...posts.map(
      (post) => `  <url>
    <loc>${post.canonicalUrl}</loc>
    <lastmod>${post.date}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.8</priority>
  </url>`
    ),
  ].join('\n');

  const sitemapXml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${sitemapUrls}
</urlset>
`;
  fs.writeFileSync(path.join(publicDir, 'sitemap.xml'), sitemapXml, 'utf8');

  const robotsTxt = `User-agent: *
Allow: /

Sitemap: ${SITE_URL}/sitemap.xml
`;
  fs.writeFileSync(path.join(publicDir, 'robots.txt'), robotsTxt, 'utf8');

  console.log(`Generated public/rss.xml, public/sitemap.xml, public/robots.txt for ${posts.length} articles.`);
  console.log(`Exported ${posts.length} Dev.to markdown files -> syndication/devto/*.md`);
  console.log(`Exported ${posts.length} Medium HTML files     -> syndication/medium/*.html`);

  // Optional CLI flag: --publish-devto <slug>
  const publishIdx = process.argv.indexOf('--publish-devto');
  if (publishIdx !== -1) {
    const targetSlug = process.argv[publishIdx + 1];
    const apiKey = process.env.DEVTO_API_KEY;
    if (!apiKey) {
      console.error('Error: Set DEVTO_API_KEY environment variable to publish directly to Dev.to API.');
      process.exit(1);
    }
    const targetPost = posts.find((p) => p.slug === targetSlug);
    if (!targetPost) {
      console.error(`Error: Post slug "${targetSlug}" not found.`);
      process.exit(1);
    }
    const md = fs.readFileSync(path.join(devtoOutDir, `${targetPost.slug}.md`), 'utf8');
    const created = await publishToDevto(targetPost, md, apiKey);
    console.log(`Published draft to Dev.to: ${created.url || created.id}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
