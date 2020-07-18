export class Animate {
  public duration: number;
  public delay: number;
  public easing: (t: number, b: number, c: number, d: number) => number;
  public target: any;

  constructor(
    properties: any,
    duration: number,
    delay: number,
    easing: (t: number, b: number, c: number, d: number) => number
  ) {
    this.target = properties;
    this.duration = duration;
    this.delay = delay;
    this.easing = easing;
  }
}
