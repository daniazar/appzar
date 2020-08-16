import { Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { CanvasExplosion } from '@app/home/canvas-explosion';
import { Grid } from '@app/home/grid/Grid';
import { environment } from '@env/environment';
import { Observable, Observer } from 'rxjs';

@Component({
  selector: 'app-about',
  templateUrl: './about.component.html',
  styleUrls: ['./about.component.scss'],
})
export class AboutComponent implements OnInit {
  version: string | null = environment.version;

  quote: string | undefined;
  isLoading = false;
  ctx: CanvasRenderingContext2D;
  base64Image: string;
  @ViewChild('logo', { static: true }) public png: ElementRef<HTMLImageElement>;
  @ViewChild('canvasLogo', { static: true }) public canvas: ElementRef<HTMLCanvasElement>;
  @ViewChild('canvasGrid', { static: true }) public canvasGrid: ElementRef<HTMLCanvasElement>;

  ngOnInit() {
    this.isLoading = true;

    const imageUrl = 'assets/portrait.png';
    console.log('loading image');

    this.getBase64ImageFromURL(imageUrl).subscribe((base64data: string) => {
      console.log(base64data);
      this.base64Image = 'data:image/jpg;base64,' + base64data;
    });
  }

  getBase64ImageFromURL(url: string) {
    return Observable.create((observer: Observer<string>) => {
      const img = new Image();
      img.crossOrigin = 'Anonymous';
      img.src = url;
      if (!img.complete) {
        img.onload = () => {
          console.log(img);
          this.explosionInit(img);
          observer.complete();
        };
        img.onerror = (err) => {
          observer.error(err);
        };
      } else {
        this.explosionInit(img);
        observer.complete();
      }
    });
  }

  explosionInit(img: HTMLImageElement) {
    const canvasExp = new CanvasExplosion(this.canvas.nativeElement, img);
    const gridAnimation = Grid.DefaultGrid(this.canvasGrid.nativeElement);
    // const gridAnimation = new GridAnimation(this.canvasGrid.nativeElement);
    // gridAnimation.play();
  }
}
