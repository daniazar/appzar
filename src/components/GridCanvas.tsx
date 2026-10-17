'use client';

import React, { useEffect, useRef } from 'react';

type Direction = 'up' | 'down' | 'left' | 'right';

interface Line {
  initialDirection: Direction;
  currentDirection: Direction;
  currentColor: string;
  coordinates: [number, number][];
}

const ALLOWED_DIRECTIONS: { [key in Direction]: Direction[] } = {
  up: ['up', 'left', 'right'],
  down: ['down', 'left', 'right'],
  left: ['left', 'up', 'down'],
  right: ['right', 'up', 'down'],
};

const NEON_COLORS = [
  '#7400b8',
  '#6930c3',
  '#5e60ce',
  '#5390d9',
  '#4ea8de',
  '#48bfe3',
  '#56cfe1',
  '#64dfdf',
  '#72efdd',
  '#80ffdb',
];

export default function GridCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    if (!context) return;

    const pixelRatio = window.devicePixelRatio || 1;
    const speed = 12;
    const squareSize = 28;
    const maxLineLength = 140;
    const lineWidth = 3;
    const gridColor = 'rgba(78, 168, 222, 0.10)';
    const lines: Line[] = [];
    let animFrameId = 0;

    const getSquareLength = () => squareSize * pixelRatio;

    const pickDirection = (available: Direction[] = ['down', 'left', 'right', 'up']): Direction => {
      return available[Math.floor(Math.random() * available.length)];
    };

    const pickLineColor = (): string => {
      return NEON_COLORS[Math.floor(Math.random() * NEON_COLORS.length)];
    };

    const updateCanvasSize = () => {
      const nextWidth = canvas.offsetWidth * pixelRatio;
      const nextHeight = canvas.offsetHeight * pixelRatio;
      if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
        canvas.width = nextWidth;
        canvas.height = nextHeight;
      }
    };

    const limitLineLength = (line: Line) => {
      const calculateSum = () =>
        line.coordinates.reduce((acc, [x, y], index, arr) => {
          if (index === 0) return acc;
          return acc + Math.hypot(x - arr[index - 1][0], y - arr[index - 1][1]);
        }, 0);

      while (calculateSum() > maxLineLength * pixelRatio && line.coordinates.length > 1) {
        line.coordinates.shift();
      }
    };

    const getNewCoordinates = (line: Line, step: number = speed): void => {
      const { coordinates, currentDirection, initialDirection } = line;
      const mostRecentCoordinate = coordinates[coordinates.length - 1];
      const squareLength = getSquareLength();
      let nextNodePosition = 0;
      let lineDistancePastNextNodePosition = 0;
      let nextLinePosition = 0;

      const handleLineIntersectingNode = (currDir: Direction, oppositeDir: Direction) => {
        if (currDir === 'up' || currDir === 'down') {
          coordinates.push([mostRecentCoordinate[0], nextNodePosition]);
        } else {
          coordinates.push([nextNodePosition, mostRecentCoordinate[1]]);
        }
        line.currentDirection = pickDirection(
          ALLOWED_DIRECTIONS[initialDirection].filter((d) => d !== oppositeDir)
        );
        getNewCoordinates(line, lineDistancePastNextNodePosition);
      };

      switch (currentDirection) {
        case 'up':
          nextNodePosition = (Math.ceil(mostRecentCoordinate[1] / squareLength) - 1) * squareLength;
          nextLinePosition = mostRecentCoordinate[1] - step;
          lineDistancePastNextNodePosition = nextNodePosition - nextLinePosition;
          if (lineDistancePastNextNodePosition > 0) {
            handleLineIntersectingNode('up', 'down');
            return;
          }
          coordinates.push([mostRecentCoordinate[0], nextLinePosition]);
          return;
        case 'down':
          nextNodePosition = (Math.floor(mostRecentCoordinate[1] / squareLength) + 1) * squareLength;
          nextLinePosition = mostRecentCoordinate[1] + step;
          lineDistancePastNextNodePosition = nextLinePosition - nextNodePosition;
          if (lineDistancePastNextNodePosition > 0) {
            handleLineIntersectingNode('down', 'up');
            return;
          }
          coordinates.push([mostRecentCoordinate[0], nextLinePosition]);
          return;
        case 'left':
          nextNodePosition = (Math.ceil(mostRecentCoordinate[0] / squareLength) - 1) * squareLength;
          nextLinePosition = mostRecentCoordinate[0] - step;
          lineDistancePastNextNodePosition = nextNodePosition - nextLinePosition;
          if (lineDistancePastNextNodePosition > 0) {
            handleLineIntersectingNode('left', 'right');
            return;
          }
          coordinates.push([nextLinePosition, mostRecentCoordinate[1]]);
          return;
        case 'right':
          nextNodePosition = (Math.floor(mostRecentCoordinate[0] / squareLength) + 1) * squareLength;
          nextLinePosition = mostRecentCoordinate[0] + step;
          lineDistancePastNextNodePosition = nextLinePosition - nextNodePosition;
          if (lineDistancePastNextNodePosition > 0) {
            handleLineIntersectingNode('right', 'left');
            return;
          }
          coordinates.push([nextLinePosition, mostRecentCoordinate[1]]);
          return;
      }
    };

    const spawnLineAt = (x: number, y: number) => {
      if (lines.length > 45) return;
      const direction = pickDirection();
      lines.push({
        coordinates: [[x, y]],
        currentColor: pickLineColor(),
        initialDirection: direction,
        currentDirection: direction,
      });
    };

    let lastMouseSpawn = 0;
    const onMouseMove = (event: MouseEvent) => {
      const now = performance.now();
      if (now - lastMouseSpawn < 24) return;
      lastMouseSpawn = now;

      const rect = canvas.getBoundingClientRect();
      if (
        event.clientX < rect.left ||
        event.clientX > rect.right ||
        event.clientY < rect.top ||
        event.clientY > rect.bottom
      ) {
        return;
      }

      const squareLength = getSquareLength();
      const relX = (event.clientX - rect.left) * pixelRatio;
      const relY = (event.clientY - rect.top) * pixelRatio;
      const x = Math.round(relX / squareLength) * squareLength;
      const y = Math.round(relY / squareLength) * squareLength;
      spawnLineAt(x, y);
    };

    let isVisible = true;

    // Ambient pulse timer so the grid is alive even without mouse movement
    const ambientInterval = window.setInterval(() => {
      if (document.hidden || !isVisible || !canvas.width || !canvas.height) return;
      const squareLength = getSquareLength();
      const cols = Math.floor(canvas.width / squareLength);
      const rows = Math.floor(canvas.height / squareLength);
      if (cols > 0 && rows > 0) {
        const randX = Math.floor(Math.random() * cols) * squareLength;
        const randY = Math.floor(Math.random() * rows) * squareLength;
        spawnLineAt(randX, randY);
      }
    }, 450);

    const draw = () => {
      if (!isVisible || document.hidden) {
        animFrameId = 0;
        return;
      }
      updateCanvasSize();
      const squareLength = getSquareLength();
      const numColumns = Math.ceil(canvas.width / squareLength);
      const numRows = Math.ceil(canvas.height / squareLength);

      context.clearRect(0, 0, canvas.width, canvas.height);

      context.beginPath();
      for (let col = 0; col <= numColumns; col += 1) {
        context.moveTo(col * squareLength, 0);
        context.lineTo(col * squareLength, canvas.height);
      }
      for (let row = 0; row <= numRows; row += 1) {
        context.moveTo(0, row * squareLength);
        context.lineTo(canvas.width, row * squareLength);
      }
      context.strokeStyle = gridColor;
      context.lineWidth = 1;
      context.stroke();

      for (let i = lines.length - 1; i >= 0; i--) {
        const line = lines[i];
        const { coordinates: lineCoords, currentColor } = line;

        context.save();
        context.beginPath();
        context.moveTo(lineCoords[0][0], lineCoords[0][1]);
        for (const coord of lineCoords) {
          context.lineTo(coord[0], coord[1]);
        }
        getNewCoordinates(line);
        limitLineLength(line);

        context.lineWidth = lineWidth * pixelRatio;
        context.lineCap = 'round';
        context.lineJoin = 'round';
        context.shadowColor = currentColor;
        context.shadowBlur = 10;
        context.strokeStyle = currentColor;
        context.stroke();
        context.restore();

        const lastCoord = lineCoords[lineCoords.length - 1];
        if (
          !lastCoord ||
          lastCoord[0] >= canvas.width ||
          lastCoord[0] <= 0 ||
          lastCoord[1] >= canvas.height ||
          lastCoord[1] <= 0
        ) {
          lines.splice(i, 1);
        }
      }

      animFrameId = window.requestAnimationFrame(draw);
    };

    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (!entry) return;
        isVisible = entry.isIntersecting;
        if (isVisible && !animFrameId) {
          animFrameId = window.requestAnimationFrame(draw);
        }
      },
      { threshold: 0.01 }
    );
    observer.observe(canvas);

    window.addEventListener('mousemove', onMouseMove, { passive: true });
    animFrameId = window.requestAnimationFrame(draw);

    return () => {
      observer.disconnect();
      window.removeEventListener('mousemove', onMouseMove);
      window.clearInterval(ambientInterval);
      if (animFrameId) window.cancelAnimationFrame(animFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 h-full w-full opacity-90"
    />
  );
}
