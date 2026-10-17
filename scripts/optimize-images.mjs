import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import sharp from 'sharp';

const rootDir = process.cwd();
const postsDir = path.join(rootDir, 'content', 'posts');
const publicDir = path.join(rootDir, 'public');
const coversDir = path.join(publicDir, 'images', 'covers');
const thumbsDir = path.join(publicDir, 'images', 'thumbs');

if (!fs.existsSync(thumbsDir)) {
  fs.mkdirSync(thumbsDir, { recursive: true });
}

function normalizeRelPath(rawPath) {
  if (!rawPath) return null;
  const cleaned = rawPath.trim().replace(/^(\.\.\/|\.\/|\/)+/, '');
  if (cleaned.startsWith('http://') || cleaned.startsWith('https://')) return null;
  return cleaned;
}

function extractFirstImage(markdown) {
  const match = markdown.match(/!\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)/);
  if (match && match[1]) {
    return normalizeRelPath(match[1]);
  }
  return null;
}

async function main() {
  let totalBeforeBytes = 0;
  let totalCoverAfterBytes = 0;
  let totalThumbAfterBytes = 0;
  let processedCovers = 0;
  let processedThumbs = 0;

  // 1. Optimize all cover images in public/images/covers/*.png -> .webp + compressed .png
  const coverFiles = fs.existsSync(coversDir)
    ? fs.readdirSync(coversDir).filter((f) => /\.(png|jpg|jpeg)$/i.test(f))
    : [];

  for (const file of coverFiles) {
    const fullPath = path.join(coversDir, file);
    const baseName = file.replace(/\.(png|jpg|jpeg)$/i, '');
    const webpCoverPath = path.join(coversDir, `${baseName}.webp`);
    const thumbWebpPath = path.join(thumbsDir, `${baseName}.webp`);

    const statBefore = fs.statSync(fullPath);
    totalBeforeBytes += statBefore.size;

    const inputBuffer = fs.readFileSync(fullPath);

    // High-res optimized WebP cover (max width 1280, quality 84)
    await sharp(inputBuffer)
      .resize({ width: 1280, withoutEnlargement: true })
      .webp({ quality: 84, effort: 5 })
      .toFile(webpCoverPath);

    // Also compress the fallback .png/.jpg in-place as progressive JPEG/PNG so legacy references stay small
    const optimizedFallback = await sharp(inputBuffer)
      .resize({ width: 1280, withoutEnlargement: true })
      .jpeg({ quality: 84, mozjpeg: true })
      .toBuffer();
    fs.writeFileSync(fullPath, optimizedFallback);

    // Dedicated 640x360 WebP thumbnail for card grids (quality 78)
    await sharp(inputBuffer)
      .resize(640, 360, { fit: 'cover', position: 'center' })
      .webp({ quality: 78, effort: 5 })
      .toFile(thumbWebpPath);

    totalCoverAfterBytes += fs.statSync(webpCoverPath).size;
    totalThumbAfterBytes += fs.statSync(thumbWebpPath).size;
    processedCovers++;
    processedThumbs++;
  }

  // 2. Ensure every blog post without a dedicated cover in public/images/covers also gets a dedicated 640x360 thumbnail in public/images/thumbs/<slug>.webp
  const postFiles = fs.readdirSync(postsDir).filter((f) => f.endsWith('.md'));
  for (const file of postFiles) {
    const slug = file.replace(/\.md$/, '');
    const thumbWebpPath = path.join(thumbsDir, `${slug}.webp`);
    if (fs.existsSync(thumbWebpPath)) continue;

    const mdContent = fs.readFileSync(path.join(postsDir, file), 'utf8');
    const { data, content } = matter(mdContent);
    const relImg =
      normalizeRelPath(data.coverImage || data.thumb_img_path || data.thumbImg || data.content_img_path) ||
      extractFirstImage(content);

    if (!relImg) continue;
    const srcImgPath = path.join(publicDir, relImg);
    if (!fs.existsSync(srcImgPath)) continue;

    const statBefore = fs.statSync(srcImgPath);
    totalBeforeBytes += statBefore.size;

    await sharp(srcImgPath)
      .resize(640, 360, { fit: 'cover', position: 'center' })
      .webp({ quality: 80, effort: 5 })
      .toFile(thumbWebpPath);

    totalThumbAfterBytes += fs.statSync(thumbWebpPath).size;
    processedThumbs++;
  }

  console.log(
    `Optimized ${processedCovers} cover images and generated ${processedThumbs} dedicated 640x360 WebP thumbnails.`
  );
  console.log(
    `Original cover total: ${(totalBeforeBytes / (1024 * 1024)).toFixed(2)} MB`
  );
  console.log(
    `Optimized WebP covers total: ${(totalCoverAfterBytes / (1024 * 1024)).toFixed(2)} MB`
  );
  console.log(
    `Dedicated 640x360 WebP thumbnails total: ${(totalThumbAfterBytes / (1024 * 1024)).toFixed(2)} MB (avg ${(
      totalThumbAfterBytes /
      Math.max(1, processedThumbs) /
      1024
    ).toFixed(1)} KB per thumbnail)`
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
