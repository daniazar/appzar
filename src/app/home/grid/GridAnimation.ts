import { Easing } from './Easing';
import { GridDot } from './GridDot';
import { GridLine } from './GridLine';

// Main animation class
export class GridAnimation {
  public canvas: HTMLCanvasElement;
  public ratio: number;
  public size: number;
  public duration: number;
  public dotSize: number;
  public lineColor: string;
  public dotColor: string;
  public lineEasing: any;
  public dotEasing: any;
  public lines: any[];
  public dots: any[];
  public numberOfHorizontalLines: number;
  public numberOfVerticalLines: number;
  public global_width: number;
  public global_height: number;
  public ctx: CanvasRenderingContext2D;
  public startTime: Date;

  constructor(element: HTMLCanvasElement, duration: number = 2500, size: number = 64) {
    this.canvas = element;
    this.ratio = 1;
    this.size = size;
    this.duration = duration;
    this.dotSize = 1;
    this.lineColor = 'rgba(50,80,150, .5)';
    this.dotColor = 'rgba(0, 173, 255, 1)';
    this.lineEasing = Easing.easeInOutQuint; // Easing.easeOutQuart;
    this.dotEasing = Easing.easeInOutQuad;

    this.lines = [];
    this.dots = [];
    this.numberOfHorizontalLines = undefined;
    this.numberOfVerticalLines = undefined;
    this.global_width = undefined;
    this.global_height = undefined;
    this.ctx = this.canvas.getContext('2d');

    this._update = this._update.bind(this);
    this._rescale = this._rescale.bind(this);

    this._init();
  }

  ////////////////////////////////////////////////
  // PUBLIC API
  ////////////////////////////////////////////////

  // Play animation from beginning
  play() {
    this.canvas.classList.add('js-animate');
    if (this.ctx) {
      console.log('GridAnimation.play()');
      this.startTime = new Date();
      this._draw();
      this._update();
    }
  }

  // Reset animation
  reset() {
    this._clearCanvas();
    this.canvas.classList.remove('js-animate');
  }

  // Skip to end of animation and draw last frame
  skipToEnd() {
    this.canvas.classList.add('js-animate');
    if (this.ctx) {
      this._draw(1);
    }
  }

  ////////////////////////////////////////////////
  // PRIVATE METHODS
  ////////////////////////////////////////////////
  _init() {
    if (this.ctx) {
      window.addEventListener('resize', this._rescale);
      this._rescale();
    }
  }

  _clearCanvas() {
    if (this.ctx) {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  // return animation progress as value between 0 and 1
  _animationProgress() {
    if (this.startTime === undefined) {
      this.startTime = new Date();
    }
    const elapsed = new Date().getTime() - this.startTime.getTime();
    return Math.min(1, Math.max(0, elapsed / this.duration));
  }

  /* Returns a random integer between min (inclusive) and max (inclusive)
   * Using Math.round() will give you a non-uniform distribution!
   */
  _getRandomInt(min: number, max: number) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  _buildAnimation() {
    this._buildLines();
    this._buildDots();
  }

  _buildLines() {
    // create Horizontal lines
    const delayPerHLine = this.duration / this.numberOfHorizontalLines / 2;
    for (let i = 0; i < this.numberOfHorizontalLines; i++) {
      const lineDelay = delayPerHLine * i;
      const line = new GridLine(this.ctx, 0, this.size * i);
      if (i % 2) {
        line.animateTo({ x: this.global_width }, this.duration - lineDelay, lineDelay, this.lineEasing);
      } else {
        line.x = this.global_width;
        line.animateTo({ x: 0 }, this.duration - lineDelay, lineDelay, this.lineEasing);
      }

      this.lines.push(line);
    }

    // create Vertical Lines
    const delayPerVLine = this.duration / this.numberOfVerticalLines / 2;
    for (let j = 0; j < this.numberOfVerticalLines; j++) {
      const lineDelay = delayPerVLine * j;
      const line = new GridLine(this.ctx, this.size * j, 0);
      if (j % 2) {
        line.animateTo({ y: this.global_height }, this.duration - lineDelay, lineDelay, this.lineEasing);
      } else {
        line.y = this.global_height;
        line.animateTo({ y: 0 }, this.duration - lineDelay, lineDelay, this.lineEasing);
      }
      this.lines.push(line);
    }
  }

  _buildDots() {
    const numberOfDots = this.numberOfHorizontalLines * this.numberOfVerticalLines;
    const dotDuration = this.duration * 0.3;
    const dotMinDelay = this.duration * 0.2;
    const dotMaxDelay = this.duration * 0.6;
    for (let k = 0; k < numberOfDots; k++) {
      const dotDelay = this._getRandomInt(dotMinDelay, dotMaxDelay);
      const x = this.size * (k % this.numberOfVerticalLines);
      const y = this.size * Math.floor(k / this.numberOfVerticalLines);

      const dot = new GridDot(this.ctx, x, y, 0);
      dot.animateTo({ size: this.dotSize }, dotDuration, dotDelay, this.dotEasing);

      this.dots.push(dot);
    }
  }

  _rescale() {
    this.global_width = Math.min(window.innerWidth, window.screen.width);
    this.global_height = Math.min(window.innerHeight, window.screen.height);

    this.ratio = window.devicePixelRatio || 1;

    this.canvas.setAttribute('width', (this.global_width * this.ratio).toString());
    this.canvas.setAttribute('height', (this.global_height * this.ratio).toString());

    this.lines = [];
    this.dots = [];
    this.numberOfHorizontalLines = Math.ceil(this.global_height / this.size);
    this.numberOfVerticalLines = Math.ceil(this.global_width / this.size);
    // this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    this._buildAnimation();
    this._draw();
  }

  // Draw current frame of animation
  _draw(progress: number = this._animationProgress()) {
    this.ctx.save();
    // clear previous frame
    this._clearCanvas();
    // scale up for retina
    this.ctx.scale(this.ratio, this.ratio);
    // straddle our lines to make them sharp: http://diveintohtml5.info/canvas.html#pixel-madness
    this.ctx.translate(0.5, 0.5);

    /////////////////////////////////////
    // high-dpi drawing start
    /////////////////////////////////////

    // Draw lines
    this.ctx.lineWidth = 1;
    this.ctx.strokeStyle = this.lineColor;
    for (const line of this.lines) {
      line.draw(progress);
    }

    // Draw dots
    this.ctx.fillStyle = this.dotColor;
    for (const dot of this.dots) {
      dot.draw(progress);
    }

    /////////////////////////////////////
    // high-dpi drawing end
    /////////////////////////////////////
    this.ctx.restore();
  }

  // Request animation Frame callback
  _update() {
    // only draw until animation has completed
    if (this._animationProgress() < 1) {
      requestAnimationFrame(this._update);
    }
    this._draw();
  }
}
