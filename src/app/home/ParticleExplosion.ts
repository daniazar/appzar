import { Mouse } from './Mouse';

export class ParticleExplosion {
  private x: number;
  private y: number;
  private color: string;
  private size: number;
  private baseX: number;
  private baseY: number;
  private density: number;
  private context: CanvasRenderingContext2D;

  // tslint:disable-next-line:max-line-length
  constructor(
    x: number,
    y: number,
    color: string,
    size: number,
    canvas: HTMLCanvasElement,
    context: CanvasRenderingContext2D,
    png: HTMLImageElement
  ) {
    this.x = x + canvas.width - png.width; // * 2,
    this.y = y + canvas.height - png.height; // * 2,
    (this.color = color),
      (this.size = 2),
      (this.baseX = x + canvas.width / 2 - png.width * 2),
      (this.baseY = y + canvas.height / 2 - png.height * 2),
      (this.density = Math.random() * 10 + 2);
    this.context = context;
  }
  draw() {
    this.context.beginPath();
    this.context.arc(this.x, this.y, this.size, 0, Math.PI * 2);
    this.context.closePath();
    this.context.fill();
  }
  update() {
    this.context.fillStyle = this.color;
    // check mouse position/particle position - collision detection
    let dx = Mouse.x - this.x;
    let dy = Mouse.y - this.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    const forceDirectionX = dx / distance;
    const forceDirectionY = dy / distance;
    // distance past which the force is zero
    const maxDistance = 100;
    let force = (maxDistance - distance) / maxDistance;

    // if we go below zero, set it to zero.
    if (force < 0) {
      force = 0;
    }

    const directionX = forceDirectionX * force * this.density * 0.9;
    const directionY = forceDirectionY * force * this.density * 0.9;

    if (distance < Mouse.radius + this.size) {
      this.x -= directionX;
      this.y -= directionY;
    } else {
      if (this.x !== this.baseX) {
        dx = this.x - this.baseX;
        dy = this.y - this.baseY;
        this.x -= dx / 5;
      }
      if (this.y !== this.baseY) {
        dx = this.x - this.baseX;
        dy = this.y - this.baseY;
        this.y -= dy / 5;
      }
    }
    this.draw();
  }
}
