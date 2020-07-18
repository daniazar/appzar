import { Animable } from './Animable';

export class GridLine extends Animable {
  public ctx: CanvasRenderingContext2D;
  public x: number;
  public y: number;
  constructor(ctx: CanvasRenderingContext2D, x: number, y: number) {
    super();
    this.ctx = ctx;
    this.x = x;
    this.y = y;
  }

  draw(progress: number) {
    this.ctx.beginPath();
    this.ctx.moveTo(this.x, this.y);

    const x = this._getAnimatedPropery('x', progress);
    const y = this._getAnimatedPropery('y', progress);

    this.ctx.lineTo(x, y);
    this.ctx.stroke();
  }
}
