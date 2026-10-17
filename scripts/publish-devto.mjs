/**
 * Automated Dev.to Publisher & Sync Script
 *
 * Reads the pre-formatted syndication Markdown files in `syndication/devto/*.md`
 * (which already contain `canonical_url: https://www.appzar.com.ar/posts/<slug>/`
 * and absolute image URLs) and creates or updates them on Dev.to via the Forem API.
 *
 * Usage:
 *   # Publish a single article as a draft on Dev.to:
 *   $env:DEVTO_API_KEY="your_key"; node scripts/publish-devto.mjs --slug isurfer-gpu-raytracing-algebraic-surfaces-ios
 *
 *   # Publish a single article live:
 *   $env:DEVTO_API_KEY="your_key"; node scripts/publish-devto.mjs --slug isurfer-gpu-raytracing-algebraic-surfaces-ios --publish
 *
 *   # Sync all articles as drafts (with automatic rate-limit spacing):
 *   $env:DEVTO_API_KEY="your_key"; node scripts/publish-devto.mjs --all
 */

import fs from 'node:fs';
import path from 'node:path';
import matter from 'gray-matter';

const DEVTO_DIR = path.join(process.cwd(), 'syndication', 'devto');
const API_KEY = process.env.DEVTO_API_KEY;

function parseArgs(argv) {
  const args = {
    slug: null,
    all: false,
    publish: false,
    dryRun: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--slug' && argv[i + 1]) {
      args.slug = argv[++i];
    } else if (arg.startsWith('--slug=')) {
      args.slug = arg.slice('--slug='.length);
    } else if (arg === '--all') {
      args.all = true;
    } else if (arg === '--publish') {
      args.publish = true;
    } else if (arg === '--dry-run') {
      args.dryRun = true;
    }
  }
  return args;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchExistingDevtoArticles() {
  const res = await fetch('https://dev.to/api/articles/me/all?per_page=1000', {
    headers: {
      'api-key': API_KEY,
      Accept: 'application/vnd.forem.api-v1+json',
    },
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch existing Dev.to articles: HTTP ${res.status}`);
  }
  return res.json();
}

async function upsertDevtoArticle(filePath, existingByCanonical, publishFlag, dryRun) {
  const raw = fs.readFileSync(filePath, 'utf8');
  const parsed = matter(raw);
  const canonicalUrl = parsed.data.canonical_url;
  const title = parsed.data.title;

  // Override `published` in frontmatter if `--publish` was explicitly passed
  const updatedFrontmatter = {
    ...parsed.data,
    published: publishFlag ? true : Boolean(parsed.data.published),
  };
  const bodyMarkdown = matter.stringify(parsed.content, updatedFrontmatter);

  const existing = existingByCanonical.get(canonicalUrl);

  if (dryRun) {
    console.log(
      `[DRY-RUN] Would ${existing ? `UPDATE (#${existing.id})` : 'CREATE'} "${title}" (published=${updatedFrontmatter.published}) -> ${canonicalUrl}`
    );
    return;
  }

  const url = existing
    ? `https://dev.to/api/articles/${existing.id}`
    : 'https://dev.to/api/articles';
  const method = existing ? 'PUT' : 'POST';

  const res = await fetch(url, {
    method,
    headers: {
      'Content-Type': 'application/json',
      'api-key': API_KEY,
      Accept: 'application/vnd.forem.api-v1+json',
    },
    body: JSON.stringify({
      article: {
        body_markdown: bodyMarkdown,
      },
    }),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Dev.to API ${method} failed (HTTP ${res.status}): ${errText}`);
  }

  const data = await res.json();
  console.log(
    `${existing ? 'Updated' : 'Created'} Dev.to article: "${data.title}" -> ${data.url}`
  );
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (!args.slug && !args.all) {
    console.log('Usage:');
    console.log('  node scripts/publish-devto.mjs --slug <post-slug> [--publish] [--dry-run]');
    console.log('  node scripts/publish-devto.mjs --all [--publish] [--dry-run]');
    process.exit(0);
  }

  if (!API_KEY && !args.dryRun) {
    console.error('Error: DEVTO_API_KEY environment variable is required (or pass --dry-run).');
    process.exit(1);
  }

  if (!fs.existsSync(DEVTO_DIR)) {
    console.error('Error: syndication/devto directory not found. Run `npm run syndicate` first.');
    process.exit(1);
  }

  const files = args.slug
    ? [path.join(DEVTO_DIR, `${args.slug}.md`)]
    : fs
        .readdirSync(DEVTO_DIR)
        .filter((f) => f.endsWith('.md'))
        .map((f) => path.join(DEVTO_DIR, f));

  for (const f of files) {
    if (!fs.existsSync(f)) {
      console.error(`File not found: ${f}`);
      process.exit(1);
    }
  }

  const existingByCanonical = new Map();
  if (!args.dryRun) {
    const existingList = await fetchExistingDevtoArticles();
    for (const item of existingList) {
      if (item.canonical_url) {
        existingByCanonical.set(item.canonical_url, item);
      }
    }
  }

  for (let i = 0; i < files.length; i++) {
    await upsertDevtoArticle(files[i], existingByCanonical, args.publish, args.dryRun);
    if (i < files.length - 1 && !args.dryRun) {
      // Respect Dev.to API rate limit (35s between article creations)
      await sleep(3500);
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
