import { NgModule } from '@angular/core';
import { PreloadAllModules, RouterModule, Routes } from '@angular/router';
import { Shell } from '@app/shell/shell.service';

const routes: Routes = [
  Shell.childRoutes([
    { path: 'about', loadChildren: () => import('./about/about.module').then((m) => m.AboutModule) },
    {
      path: 'experiments',
      loadChildren: () => import('./experiments/experiments.module').then((m) => m.ExperimentsModule),
    },
    { path: 'projects', loadChildren: () => import('./projects/projects.module').then((m) => m.ProjectsModule) },
  ]),
  // Fallback when no prior route is matched
  { path: 'blog', redirectTo: 'blog.appzar.com.ar', pathMatch: 'full' },

  { path: '**', redirectTo: '', pathMatch: 'full' },
];

@NgModule({
  imports: [RouterModule.forRoot(routes, { preloadingStrategy: PreloadAllModules })],
  exports: [RouterModule],
  providers: [],
})
export class AppRoutingModule {}
