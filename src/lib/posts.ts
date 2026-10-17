import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

export interface BlogPost {
  slug: string;
  title: string;
  subtitle?: string;
  excerpt: string;
  date: string;
  thumbImg?: string;
  contentImg?: string;
  tags: string[];
  readingTime: string;
  content: string;
  seriesOrder?: number;
}

export type BlogPostSummary = Omit<BlogPost, 'content'>;

const postsDirectory = path.join(process.cwd(), 'content/posts');

export function normalizeAssetPath(rawPath?: string): string | undefined {
  if (!rawPath) return undefined;
  const cleaned = rawPath.trim().replace(/^(\.\.\/|\.\/)+/, '');
  const normalized =
    cleaned.startsWith('http://') || cleaned.startsWith('https://') || cleaned.startsWith('/')
      ? cleaned
      : `/${cleaned}`;

  if (normalized.startsWith('/') && /\.(png|jpg|jpeg)$/i.test(normalized)) {
    const webpCandidate = normalized.replace(/\.(png|jpg|jpeg)$/i, '.webp');
    const diskWebp = path.join(process.cwd(), 'public', webpCandidate.replace(/^\//, ''));
    if (fs.existsSync(diskWebp)) {
      return webpCandidate;
    }
  }
  return normalized;
}

function extractTitleAndStripH1(rawContent: string, frontmatterTitle?: string): { title: string; body: string } {
  const lines = rawContent.split(/\r?\n/);
  let title = frontmatterTitle || '';
  let startIdx = 0;

  // Find first non-empty line
  while (startIdx < lines.length && lines[startIdx].trim() === '') {
    startIdx++;
  }

  if (startIdx < lines.length && lines[startIdx].startsWith('# ')) {
    const h1Text = lines[startIdx].replace(/^#\s+/, '').trim();
    if (!title) {
      title = h1Text;
    }
    lines.splice(startIdx, 1);
  }

  return {
    title: title || 'Untitled Post',
    body: lines.join('\n').trim(),
  };
}

function extractFirstImage(markdown: string): string | undefined {
  const match = markdown.match(/!\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)/);
  if (match && match[1]) {
    return normalizeAssetPath(match[1]);
  }
  return undefined;
}

function extractCleanExcerpt(markdown: string, fallbackTitle: string): string {
  const lines = markdown.split(/\r?\n/);
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;
    // Skip headers, horizontal rules, images, code fences, tables, blockquotes, metadata lines, list items
    if (
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
      ) ||
      /^\*By\s/i.test(line) ||
      /^\*Hardware/i.test(line) ||
      /^\*Target/i.test(line)
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
      return plain.length > 210 ? `${plain.slice(0, 207)}...` : plain;
    }
  }
  return fallbackTitle;
}

function resolvePostDate(slug: string, markdown: string, frontmatterDate?: string): { date: string; seriesOrder: number } {
  if (frontmatterDate) {
    return { date: String(frontmatterDate), seriesOrder: 0 };
  }

  // Check if slug starts with a 2-digit part number: 01_... to 23_...
  const numMatch = slug.match(/^(\d{2})_/);
  if (numMatch) {
    const partNum = parseInt(numMatch[1], 10);
    // Spread Parts 1..23 cleanly across Aug 29, 2026 -> Oct 02, 2026
    const baseDate = new Date(Date.UTC(2026, 7, 28 + partNum)); // Aug 29 + partNum
    const yyyy = baseDate.getUTCFullYear();
    const mm = String(baseDate.getUTCMonth() + 1).padStart(2, '0');
    const dd = String(baseDate.getUTCDate()).padStart(2, '0');
    return { date: `${yyyy}-${mm}-${dd}`, seriesOrder: 100 + partNum };
  }

  // Check for engineering_blog_* series
  if (slug.startsWith('engineering_blog_')) {
    if (slug.includes('webgpu') || slug.includes('piper')) {
      return { date: '2026-10-05', seriesOrder: 200 };
    }
    const partMatch = slug.match(/part_(\d+)/);
    const p = partMatch ? parseInt(partMatch[1], 10) : 1;
    return { date: `2026-09-${String(10 + p).padStart(2, '0')}`, seriesOrder: 50 + p };
  }

  // Try parsing inline **Date**: Month Day, Year
  const dateLineMatch = markdown.match(/\*\*Date\*\*:\s*([^\r\n]+)/i);
  if (dateLineMatch) {
    const parsed = new Date(dateLineMatch[1].replace(/\*+/g, '').trim());
    if (!Number.isNaN(parsed.getTime())) {
      return { date: parsed.toISOString().slice(0, 10), seriesOrder: 10 };
    }
  }

  return { date: '2026-09-15', seriesOrder: 1 };
}

function inferTags(slug: string, title: string, content: string, explicitTags?: string[]): string[] {
  const explicitJoined = (explicitTags || []).join(' ');
  const combined = `${slug} ${title} ${explicitJoined} ${content.slice(0, 1500)}`.toLowerCase();
  const tags: string[] = [];

  const isRubyNewsStudio = /^\d{2}_/.test(slug) || slug.startsWith('engineering_blog_');

  if (
    combined.includes('tensorrt') ||
    combined.includes('cuda') ||
    combined.includes('blackwell') ||
    combined.includes('fp8') ||
    combined.includes('triton') ||
    combined.includes('nvenc') ||
    combined.includes('onnx')
  ) {
    tags.push('CUDA & TensorRT');
  }
  if (
    combined.includes('musetalk') ||
    combined.includes('ditto') ||
    combined.includes('talking-head') ||
    combined.includes('talking head') ||
    combined.includes('lip-sync') ||
    combined.includes('rife') ||
    combined.includes('inpainting') ||
    slug.includes('cinezero') ||
    combined.includes('veo 3') ||
    combined.includes('google flow')
  ) {
    tags.push('Talking Head & Video AI');
  }
  if (
    combined.includes('tts') ||
    combined.includes('omnivoice') ||
    combined.includes('piper') ||
    combined.includes('vocoder') ||
    combined.includes('prosody') ||
    combined.includes('voice cloning') ||
    combined.includes('speculative vad') ||
    slug.includes('forensic_tts') ||
    slug.includes('frontier_dsp') ||
    slug.includes('omnivoice') ||
    slug.includes('eliminating_robotic_audio')
  ) {
    tags.push('Voice & TTS');
  }
  if (
    combined.includes('webgpu') ||
    combined.includes('webgl') ||
    combined.includes('opengl') ||
    combined.includes('glsl') ||
    combined.includes('raytracing') ||
    combined.includes('ray tracing') ||
    slug.includes('isurfer') ||
    slug.includes('surfzar') ||
    slug.includes('lsystemts') ||
    slug.includes('unity') ||
    combined.includes('unity 3d') ||
    combined.includes('three.js')
  ) {
    tags.push('Computer Graphics');
  }
  if (
    isRubyNewsStudio ||
    combined.includes('ai agent') ||
    combined.includes('agentic') ||
    slug.includes('jules') ||
    slug.includes('mcp') ||
    slug.includes('molten-ai') ||
    combined.includes('genkit') ||
    slug.includes('cinezero') ||
    slug.includes('architectura-ai') ||
    slug.includes('contractapp') ||
    slug.includes('cloudcam')
  ) {
    tags.push('AI & Agents');
  }
  if (
    combined.includes('azure') ||
    slug.includes('auto-scale') ||
    slug.includes('ssl') ||
    slug.includes('moltenminds') ||
    slug.includes('firewall') ||
    slug.includes('steganography')
  ) {
    tags.push('Cloud & Security');
  }
  if (
    slug.includes('surfzar') ||
    slug.includes('cinezero') ||
    slug.includes('architectura-ai') ||
    slug.includes('pdfmaster') ||
    slug.includes('contractapp') ||
    slug.includes('molten-ai') ||
    slug.includes('discord-schedule-bot') ||
    slug.includes('inmoflow') ||
    slug.includes('webscreen') ||
    slug.includes('lsystemts') ||
    slug.includes('google-flow-mcp') ||
    slug.includes('electron') ||
    slug.includes('angular') ||
    slug.includes('fixing-old-images') ||
    slug.includes('piper_tts_in_the_browser') ||
    slug.includes('webgpu_browser_native')
  ) {
    tags.push('TypeScript & Web');
  }
  if (
    slug.includes('teshis') ||
    slug.includes('aoe2') ||
    slug.includes('cloudcam') ||
    slug.includes('isurfer') ||
    slug.includes('steganography') ||
    slug.includes('firewall')
  ) {
    tags.push('Systems & Desktop');
  }

  return tags.length > 0 ? tags : ['TypeScript & Web'];
}

function calculateReadingTime(content: string): string {
  const words = content.trim().split(/\s+/).length;
  const minutes = Math.max(1, Math.ceil(words / 220));
  return `${minutes} min read`;
}

export function getAllPosts(): BlogPost[] {
  if (!fs.existsSync(postsDirectory)) {
    return [];
  }

  const fileNames = fs.readdirSync(postsDirectory).filter((file) => file.endsWith('.md'));

  const allPostsData: BlogPost[] = fileNames.map((fileName) => {
    const slug = fileName.replace(/\.md$/, '');
    const fullPath = path.join(postsDirectory, fileName);
    const fileContents = fs.readFileSync(fullPath, 'utf8');
    const { data, content: rawBody } = matter(fileContents);

    const { title, body } = extractTitleAndStripH1(rawBody, data.title);
    const { date, seriesOrder } = resolvePostDate(slug, rawBody, data.date);

    // Check for dedicated 640x360 WebP thumbnail and 1280p WebP/PNG cover image
    const dedicatedThumbWebpPath = path.join(process.cwd(), 'public', 'images', 'thumbs', `${slug}.webp`);
    const dedicatedCoverWebpPath = path.join(process.cwd(), 'public', 'images', 'covers', `${slug}.webp`);
    const dedicatedCoverPngPath = path.join(process.cwd(), 'public', 'images', 'covers', `${slug}.png`);

    const dedicatedThumbUrl = fs.existsSync(dedicatedThumbWebpPath)
      ? `/images/thumbs/${slug}.webp`
      : undefined;
    const dedicatedCoverUrl = fs.existsSync(dedicatedCoverWebpPath)
      ? `/images/covers/${slug}.webp`
      : fs.existsSync(dedicatedCoverPngPath)
        ? `/images/covers/${slug}.png`
        : undefined;

    const firstBodyImg = extractFirstImage(body);
    const thumbImg =
      dedicatedThumbUrl ||
      dedicatedCoverUrl ||
      normalizeAssetPath(data.thumb_img_path || data.thumbImg) ||
      firstBodyImg;
    const contentImg =
      dedicatedCoverUrl ||
      normalizeAssetPath(data.content_img_path || data.contentImg || data.thumb_img_path);

    const excerpt = data.excerpt || data.subtitle || extractCleanExcerpt(body, title);

    return {
      slug,
      title,
      subtitle: data.subtitle,
      excerpt,
      date,
      thumbImg,
      contentImg,
      tags: inferTags(slug, title, body, data.tags),
      readingTime: calculateReadingTime(body),
      content: body,
      seriesOrder,
    };
  });

  return allPostsData.sort((a, b) => {
    if (a.date !== b.date) {
      return a.date < b.date ? 1 : -1;
    }
    return (b.seriesOrder || 0) - (a.seriesOrder || 0);
  });
}

export function getAllPostSummaries(): BlogPostSummary[] {
  return getAllPosts().map(({ content: _content, ...summary }) => summary);
}

export function getPostBySlug(slug: string): BlogPost | undefined {
  const posts = getAllPosts();
  return posts.find((post) => post.slug === slug);
}
