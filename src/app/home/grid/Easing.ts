export class Easing {
  static easeInSine(t: number, b: number, c: number, d: number) {
    return -c * Math.cos((t / d) * (Math.PI / 2)) + c + b;
  }

  static easeOutSine(t: number, b: number, c: number, d: number) {
    return c * Math.sin((t / d) * (Math.PI / 2)) + b;
  }

  static easeInOutSine(t: number, b: number, c: number, d: number) {
    return (-c / 2) * (Math.cos((Math.PI * t) / d) - 1) + b;
  }

  static easeInQuad(t: number, b: number, c: number, d: number) {
    t /= d;
    return c * t * t + b;
  }

  static easeOutQuad(t: number, b: number, c: number, d: number) {
    t /= d;
    return -c * t * (t - 2) + b;
  }

  static easeInOutQuad(t: number, b: number, c: number, d: number) {
    t /= d / 2;
    if (t < 1) {
      return (c / 2) * t * t + b;
    }
    t--;
    return (-c / 2) * (t * (t - 2) - 1) + b;
  }

  static easeInOutQuart(t: number, b: number, c: number, d: number) {
    t /= d / 2;
    if (t < 1) {
      return (c / 2) * t * t * t * t + b;
    }
    t -= 2;
    return (-c / 2) * (t * t * t * t - 2) + b;
  }

  static easeInQuart(t: number, b: number, c: number, d: number) {
    t /= d;
    return c * t * t * t * t + b;
  }

  static easeOutQuart(t: number, b: number, c: number, d: number) {
    t /= d;
    t--;
    return -c * (t * t * t * t - 1) + b;
  }

  static easeOutQuint(t: number, b: number, c: number, d: number) {
    t /= d;
    t--;
    return c * (t * t * t * t * t + 1) + b;
  }

  static easeInOutQuint(t: number, b: number, c: number, d: number) {
    t /= d / 2;
    if (t < 1) {
      return (c / 2) * t * t * t * t * t + b;
    }
    t -= 2;
    return (c / 2) * (t * t * t * t * t + 2) + b;
  }
}
