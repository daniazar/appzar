import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import sharp from 'sharp';

const rootDir = process.cwd();
const postsDir = path.join(rootDir, 'content', 'posts');
const publicDir = path.join(rootDir, 'public');
const coversDir = path.join(publicDir, 'images', 'covers');
const thumbsDir = path.join(publicDir, 'images', 'thumbs');
const assetsDir = path.join(publicDir, 'assets');

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

function walkFiles(dir, out = []) {
  if (!fs.existsSync(dir)) return out;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walkFiles(full, out);
    } else {
      out.push(full);
    }
  }
  return out;
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
    const webpCoverBuffer = await sharp(inputBuffer)
      .resize({ width: 1280, withoutEnlargement: true })
      .webp({ quality: 84, effort: 5 })
      .toBuffer();
    fs.writeFileSync(webpCoverPath, webpCoverBuffer);

    // Also compress the fallback .png/.jpg in-place as progressive JPEG so legacy references stay small
    const optimizedFallback = await sharp(inputBuffer)
      .resize({ width: 1280, withoutEnlargement: true })
      .jpeg({ quality: 84, mozjpeg: true })
      .toBuffer();
    fs.writeFileSync(fullPath, optimizedFallback);

    // Dedicated 640x360 WebP thumbnail for card grids (quality 78)
    const thumbWebpBuffer = await sharp(inputBuffer)
      .resize(640, 360, { fit: 'cover', position: 'center' })
      .webp({ quality: 78, effort: 5 })
      .toBuffer();
    fs.writeFileSync(thumbWebpPath, thumbWebpBuffer);

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

    const thumbBuffer = await sharp(fs.readFileSync(srcImgPath), { animated: false })
      .resize(640, 360, { fit: 'cover', position: 'center' })
      .webp({ quality: 80, effort: 5 })
      .toBuffer();
    fs.writeFileSync(thumbWebpPath, thumbBuffer);

    totalThumbAfterBytes += fs.statSync(thumbWebpPath).size;
    processedThumbs++;
  }

  // 3. Generate dedicated 640x360 WebP thumbnails for all Featured Projects that used raw /assets/* files
  const projectThumbSources = [
    { out: 'project_surfzar.webp', src: 'assets/surfzar/studio_barth.png' },
    { out: 'project_aoe2_ui_mod.webp', src: 'assets/aoe2-ui-mod.jpg' },
    { out: 'project_image_metadata.webp', src: 'assets/image-metadata.png' },
    { out: 'project_tvmaster.webp', src: 'assets/tvmaster.png' },
    { out: 'project_cloudcam.webp', src: 'assets/cloudcam.png' },
    { out: 'project_isurfer.webp', src: 'assets/isurfer.png' },
    { out: 'project_dreamlist.webp', src: 'assets/dreamlist.png' },
    { out: 'project_stenography.webp', src: 'assets/stenography.png' },
    { out: 'project_asteroids.webp', src: 'assets/asteroids.png' },
    { out: 'project_cityracer.webp', src: 'assets/cityracer.png' },
  ];

  let projectThumbsBefore = 0;
  let projectThumbsAfter = 0;
  for (const item of projectThumbSources) {
    const srcPath = path.join(publicDir, item.src);
    const destPath = path.join(thumbsDir, item.out);
    if (!fs.existsSync(srcPath)) continue;
    projectThumbsBefore += fs.statSync(srcPath).size;
    const buf = await sharp(fs.readFileSync(srcPath))
      .resize(640, 360, { fit: 'inside', withoutEnlargement: true })
      .webp({ quality: 80, effort: 5 })
      .toBuffer();
    fs.writeFileSync(destPath, buf);
    projectThumbsAfter += buf.length;
  }

  // 4. Generate optimized .webp siblings for all inline PNG/JPG assets in public/assets/** (>40 KB)
  const assetFiles = walkFiles(assetsDir).filter((f) => /\.(png|jpg|jpeg)$/i.test(f));
  let inlineBefore = 0;
  let inlineAfter = 0;
  let inlineCount = 0;

  for (const assetPath of assetFiles) {
    const base = path.basename(assetPath).toLowerCase();
    // Keep tiny portrait.png untouched because ParticlePortrait reads raw canvas pixels from it
    if (base === 'portrait.png') continue;

    const webpSibling = assetPath.replace(/\.(png|jpg|jpeg)$/i, '.webp');
    const stat = fs.statSync(assetPath);
    if (fs.existsSync(webpSibling) && fs.statSync(webpSibling).mtimeMs >= stat.mtimeMs) {
      continue;
    }
    try {
      const buf = await sharp(fs.readFileSync(assetPath))
        .resize({ width: 1400, withoutEnlargement: true })
        .webp({ quality: 82, effort: 4 })
        .toBuffer();
      fs.writeFileSync(webpSibling, buf);
      inlineBefore += stat.size;
      inlineAfter += buf.length;
      inlineCount++;
    } catch {
      // Skip any non-standard file
    }
  }

  console.log(
    `Optimized ${processedCovers} cover images and generated ${processedThumbs} dedicated 640x360 WebP post thumbnails.`
  );
  console.log(
    `Generated ${projectThumbSources.length} dedicated project card WebP thumbnails: ${(
      projectThumbsBefore / 1024
    ).toFixed(1)} KB -> ${(projectThumbsAfter / 1024).toFixed(1)} KB`
  );
  if (inlineCount > 0) {
    console.log(
      `Converted ${inlineCount} inline blog assets in public/assets/** to WebP: ${(
        inlineBefore /
        (1024 * 1024)
      ).toFixed(2)} MB -> ${(inlineAfter / (1024 * 1024)).toFixed(2)} MB`
    );
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
