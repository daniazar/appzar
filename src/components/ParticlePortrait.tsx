'use client';

import React, { useEffect, useRef } from 'react';

interface Particle {
  x: number;
  y: number;
  baseX: number;
  baseY: number;
  color: string;
  size: number;
  density: number;
}

export default function ParticlePortrait() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animFrameId = 0;
    const particles: Particle[] = [];
    const mouse = {
      x: -9999,
      y: -9999,
      radius: 85,
    };

    const img = new Image();
    img.crossOrigin = 'Anonymous';
    img.src = '/assets/portrait.png';

    const initParticles = () => {
      const imgWidth = img.naturalWidth || img.width || 134;
      const imgHeight = img.naturalHeight || img.height || 136;

      canvas.width = imgWidth * 4;
      canvas.height = imgHeight * 4;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, imgWidth, imgHeight);
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.length = 0;

      for (let y = 0; y < imageData.height; y++) {
        for (let x = 0; x < imageData.width; x++) {
          const idx = (y * imageData.width + x) * 4;
          const alpha = imageData.data[idx + 3];
          if (alpha > 128) {
            const r = imageData.data[idx];
            const g = imageData.data[idx + 1];
            const b = imageData.data[idx + 2];
            const color = `rgba(${r},${g},${b},${alpha})`;

            const px = x * 4;
            const py = y * 4;
            const baseX = px + canvas.width / 2 - imgWidth * 2;
            const baseY = py + canvas.height / 2 - imgHeight * 2;

            particles.push({
              x: px + canvas.width - imgWidth,
              y: py + canvas.height - imgHeight,
              baseX,
              baseY,
              color,
              size: 2.1,
              density: Math.random() * 10 + 2,
            });
          }
        }
      }
    };

    const animate = () => {
      ctx.fillStyle = 'rgba(13, 17, 28, 0.28)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        let dx = mouse.x - p.x;
        let dy = mouse.y - p.y;
        const distance = Math.sqrt(dx * dx + dy * dy) || 1;
        const forceDirectionX = dx / distance;
        const forceDirectionY = dy / distance;
        const maxDistance = 100;
        let force = (maxDistance - distance) / maxDistance;
        if (force < 0) force = 0;

        const directionX = forceDirectionX * force * p.density * 0.9;
        const directionY = forceDirectionY * force * p.density * 0.9;

        if (distance < mouse.radius + p.size) {
          p.x -= directionX;
          p.y -= directionY;
        } else {
          if (p.x !== p.baseX) {
            dx = p.x - p.baseX;
            p.x -= dx / 5;
          }
          if (p.y !== p.baseY) {
            dy = p.y - p.baseY;
            p.y -= dy / 5;
          }
        }

        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.closePath();
        ctx.fill();
      }

      animFrameId = window.requestAnimationFrame(animate);
    };

    const handlePointerMove = (clientX: number, clientY: number) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = ((clientX - rect.left) / (rect.right - rect.left)) * canvas.width;
      mouse.y = ((clientY - rect.top) / (rect.bottom - rect.top)) * canvas.height;
    };

    const onMouseMove = (e: MouseEvent) => handlePointerMove(e.clientX, e.clientY);
    const onTouchMove = (e: TouchEvent) => {
      if (e.touches.length > 0) {
        handlePointerMove(e.touches[0].clientX, e.touches[0].clientY);
      }
    };
    const onPointerLeave = () => {
      mouse.x = -9999;
      mouse.y = -9999;
    };

    if (img.complete) {
      initParticles();
      animFrameId = window.requestAnimationFrame(animate);
    } else {
      img.onload = () => {
        initParticles();
        animFrameId = window.requestAnimationFrame(animate);
      };
    }

    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseleave', onPointerLeave);
    canvas.addEventListener('touchmove', onTouchMove, { passive: true });
    canvas.addEventListener('touchend', onPointerLeave);

    return () => {
      window.cancelAnimationFrame(animFrameId);
      canvas.removeEventListener('mousemove', onMouseMove);
      canvas.removeEventListener('mouseleave', onPointerLeave);
      canvas.removeEventListener('touchmove', onTouchMove);
      canvas.removeEventListener('touchend', onPointerLeave);
    };
  }, []);

  return (
    <div className="group relative inline-flex flex-col items-center">
      <div className="relative rounded-full p-1.5 bg-gradient-to-tr from-[#7400b8] via-[#4ea8de] to-[#80ffdb] shadow-[0_0_40px_-8px_rgba(78,168,222,0.55)] transition-transform duration-300 group-hover:scale-[1.03]">
        <div className="overflow-hidden rounded-full bg-[#0d111c]">
          <canvas
            ref={canvasRef}
            id="canvasLogo"
            title="Move cursor over portrait to scatter particles"
            className="h-[144px] w-[142px] cursor-crosshair rounded-full block"
          />
        </div>
      </div>
      <span className="mt-2.5 rounded-full border border-slate-800/90 bg-[#070a12]/90 px-3 py-0.5 text-[11px] font-medium tracking-wide text-cyan-300/90 shadow-md transition-opacity">
        ✦ Hover portrait to interact
      </span>
    </div>
  );
}
