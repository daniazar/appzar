import { AfterViewInit, Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { Observable, Observer } from 'rxjs';
import { finalize } from 'rxjs/operators';
import { CanvasExplosion } from './canvas-explosion';
import { Grid } from './grid/Grid';
import { QuoteService } from './quote.service';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss'],
})
export class HomeComponent implements OnInit, AfterViewInit {
  quote: string | undefined;
  isLoading = false;
  ctx: CanvasRenderingContext2D;
  base64Image: string;
  @ViewChild('logo', { static: true }) public png: ElementRef<HTMLImageElement>;
  @ViewChild('canvasLogo', { static: true }) public canvas: ElementRef<HTMLCanvasElement>;
  @ViewChild('canvasGrid', { static: true }) public canvasGrid: ElementRef<HTMLCanvasElement>;
  constructor(private quoteService: QuoteService) {}

  ngOnInit() {
    this.isLoading = true;
    this.quoteService
      .getRandomQuote({ category: 'dev' })
      .pipe(
        finalize(() => {
          this.isLoading = false;
        })
      )
      .subscribe((quote: string) => {
        this.quote = quote;
      });

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

  ngAfterViewInit() {
    // const canvas: HTMLCanvasElement = document.getElementById('canvasLogo') as HTMLCanvasElement;
    // this.ctx = canvas.getContext('2d');
    // const png = new Image();
    // png.src = 'assets/portrait.png';
    /*this.png.nativeElement.addEventListener('load', () => {
      const canvasExp = new CanvasExplosion(this.canvas.nativeElement, this.png.nativeElement);
    });*/
    /* const xmlHTTP = new XMLHttpRequest();
     xmlHTTP.open('GET', '/images/photos/badger.jpg', true);

     // Must include this line - specifies the response type we want
     xmlHTTP.responseType = 'arraybuffer';

     xmlHTTP.onload = function (e) {

       var arr = new Uint8Array(this.response);


       // Convert the int array to a binary string
       // We have to use apply() as we are converting an *array*
       // and String.fromCharCode() takes one or more single values, not
       // an array.
       var raw = String.fromCharCode.apply(null, arr);

       // This works!!!
       var b64 = btoa(raw);
       var dataURL = "data:image/jpeg;base64," + b64;
       document.getElementById("image").src = dataURL;
     };

     xmlHTTP.send();
   }*/
    /*    png.onload(() => {
        const canvasExp = new CanvasExplosion(canvas, png);
        canvasExp.drawImage();
      });*/
  }
}
