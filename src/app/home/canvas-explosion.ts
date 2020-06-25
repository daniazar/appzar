import { Mouse } from './Mouse';
import { ParticleExplosion } from './ParticleExplosion';

export class CanvasExplosion {
  private _context: CanvasRenderingContext2D;
  private _canvas: HTMLCanvasElement;
  private _png: HTMLImageElement;
  private particleArray: ParticleExplosion[] = [];
  private _rect: DOMRect;
  constructor(canvas: HTMLCanvasElement, png: HTMLImageElement) {
    this._canvas = canvas;
    this._context = canvas.getContext('2d');
    this._png = png;
    this.canvasToPixel();
    this._rect = canvas.getBoundingClientRect();

    this._canvas.addEventListener('mousemove', (event) => {
      Mouse.x = ((event.clientX - this._rect.left) / (this._rect.right - this._rect.left)) * this._canvas.width;
      Mouse.y = ((event.clientY - this._rect.top) / (this._rect.bottom - this._rect.top)) * this._canvas.height;
      // Mouse.x = (event.x + this._canvas.clientLeft - this._canvas.offsetLeft) * 4;
      // Mouse.y = (event.y + this._canvas.clientTop  - this._canvas.offsetTop) * 4;
      // console.log(Mouse.y + ' ' + event.y + ' ' + this._canvas.clientTop + ' ' + this._canvas.offset().top);
    });

    this.drawImage(this);
  }

  canvasToPixel() {
    this._canvas.width = this._png.naturalWidth * 4;
    this._canvas.height = this._png.naturalHeight * 4;
    // get mouse mouse position ///////////////////////////////
    const mouse = {
      x: 0,
      y: 0,
      radius: 20,
    };
    this._context.drawImage(this._png, 0, 0);
  }

  drawImage(explosion: CanvasExplosion) {
    const imageWidth = explosion._png.width || explosion._png.naturalWidth;
    const imageHeight = explosion._png.height || explosion._png.naturalHeight;
    const data = explosion._context.getImageData(0, 0, imageWidth, imageHeight);
    explosion._context.clearRect(0, 0, explosion._canvas.width, explosion._canvas.height);
    function init() {
      for (let y = 0, y2 = data.height; y < y2; y++) {
        for (let x = 0, x2 = data.width; x < x2; x++) {
          if (data.data[y * 4 * data.width + x * 4 + 3] > 128) {
            const positionX = x;
            const positionY = y;
            // tslint:disable-next-line:max-line-length
            const color =
              'rgb(' +
              data.data[y * 4 * data.width + x * 4] +
              ',' +
              data.data[y * 4 * data.width + x * 4 + 1] +
              ',' +
              data.data[y * 4 * data.width + x * 4 + 2] +
              ')';

            // tslint:disable-next-line:max-line-length
            explosion.particleArray.push(
              new ParticleExplosion(
                positionX * 4,
                positionY * 4,
                color,
                0,
                explosion._canvas,
                explosion._context,
                explosion._png
              )
            );
          }
        }
      }
    }
    function animate() {
      requestAnimationFrame(animate);
      explosion._context.fillStyle = 'rgba(255,255,255,.2)';
      explosion._context.fillRect(0, 0, innerWidth, innerHeight);
      // this._context.clearRect(0, 0, innerWidth, innerHeight);

      for (const particle of explosion.particleArray) {
        particle.update();
      }
    }
    init();
    animate();

    // RESIZE SETTING - empty and refill particle array every time window changes size + change canvas size
    window.addEventListener('resize', () => {
      // explosion._canvas.width = innerWidth;
      // explosion._canvas.height = innerHeight;
      explosion._context.clearRect(0, 0, explosion._canvas.width, explosion._canvas.height);
      explosion._rect = explosion._canvas.getBoundingClientRect();
    });
  }
}

// Run drawImage after page has been fully loaded
window.addEventListener('load', (event) => {
  console.log('page has loaded');
});
