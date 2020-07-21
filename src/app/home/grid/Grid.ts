import { GridConfig } from './GridConfig';
import { GridRenderer } from './GridRenderer';

export class Grid implements GridConfig {
  /** Colors used for the lines. */
  colors: string[];
  /** Speed at which the lines moves. */
  speed: number;
  /** The size of the grid squares. */
  squareSize: number;
  /** The max length a line can reach. */
  maxLineLength: number;
  /** The width of the lines.  */
  lineWidth: number;
  /** Color of the grid lines. */
  gridColor: string;

  grid: GridRenderer;

  static DefaultGrid(canvas: HTMLCanvasElement): Grid {
    const colors = [
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
    const speed = 30;
    const squareSize = 24;
    const maxLineLength = 100;
    const lineWidth = 4;
    const gridColor = 'rgba(50,80,150, .5)';

    return new Grid(colors, speed, squareSize, maxLineLength, lineWidth, gridColor, canvas);
  }

  constructor(
    colors: string[],
    speed: number,
    squareSize: number,
    maxLineLength: number,
    lineWidth: number,
    gridColor: string,
    canvas: HTMLCanvasElement
  ) {
    this.colors = colors;
    this.speed = speed;
    this.squareSize = squareSize;
    this.maxLineLength = maxLineLength;
    this.lineWidth = lineWidth;
    this.gridColor = gridColor;
    this.grid = new GridRenderer(canvas, this);
    this.grid.register();
  }
}
