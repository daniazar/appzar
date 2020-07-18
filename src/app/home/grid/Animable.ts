import { Animate } from './Animate';

export class Animable {
  public animation: Animate;
  animateTo(
    properties: any,
    duration: number,
    delay: number,
    easing: (t: number, b: number, c: number, d: number) => number
  ) {
    this.animation = new Animate(properties, duration, delay, easing);
  }
  _getAnimatedPropery(property: string, progress: number) {
    if (this.animation && typeof this.animation.target[property] !== 'undefined') {
      const delta = this.animation.target[property] - this[property];
      const duration = this.animation.duration + this.animation.delay;
      const time = Math.max(0, duration * progress - this.animation.delay);
      return this.animation.easing(time, this[property], delta, this.animation.duration);
    }
    return this[property];
  }
}
