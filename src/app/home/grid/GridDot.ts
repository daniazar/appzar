import { Animable } from './Animable';

export class GridDot extends Animable {
  public ctx: CanvasRenderingContext2D;
  public x: number;
  public y: number;
  public size: number;
  constructor(ctx: CanvasRenderingContext2D, x: number, y: number, size: number) {
    super();
    this.ctx = ctx;
    this.x = x;
    this.y = y;
    this.size = size;
  }

  draw(progress: number) {
    this.ctx.beginPath();
    this.ctx.moveTo(this.x, this.y);
    const size = this._getAnimatedPropery('size', progress);
    this.ctx.arc(this.x, this.y, size, 0, 2 * Math.PI);
    this.ctx.fill();
  }
}
