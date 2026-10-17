---
title: "SurfZar: Real-Time 3D Algebraic Surface Visualizer & Hybrid GPU Solver in WebGL2"
subtitle: "From my 2011 iPhone 3GS thesis (iSurfer) to closed-form Cardano/Ferrari root solvers, interval raymarching, dual-sided PBR shading, and watertight STL 3D printing in the browser."
excerpt: "How I evolved my 2011 iPhone GPU ray-tracing thesis (iSurfer) into SurfZar—a zero-install WebGL2 studio that renders implicit algebraic surfaces f(x,y,z)=0 at 60 FPS using analytical polynomial solvers, interval raymarching, and Marching Cubes STL export."
date: '2026-10-14'
tags: ['WebGPU & Real-Time', 'TypeScript & Web', 'Computer Graphics']
thumb_img_path: /images/covers/surfzar_realtime_3d_algebraic_surfaces_webgl.png
content_img_path: /images/covers/surfzar_realtime_3d_algebraic_surfaces_webgl.png
---

## 1. Origins: From an iPhone 3GS in 2011 to Modern WebGL2

Fifteen years ago, for my software engineering thesis at **ITBA** (*Instituto Tecnológico de Buenos Aires*), I built **iSurfer**—an interactive algebraic surface renderer for the iPhone 3GS. Inspired by the [IMAGINARY](https://www.imaginary.org/) open-mathematics project and Oberwolfach's *SURFER*, the challenge in 2011 was brutal: mobile CPUs could not evaluate millions of polynomial ray intersections per frame, so I had to write a custom ray tracer inside dynamically compiled **OpenGL ES 2.0 GLSL fragment shaders** running on the PowerVR SGX535 GPU.

**SurfZar** ([surfzar.vercel.app](https://surfzar.vercel.app)) brings that vision into the modern browser with **Next.js, React 19, Three.js (React Three Fiber), and custom WebGL2 GLSL shaders**—requiring zero installation, zero plugins, and zero backend compute.

![SurfZar 3D Studio Live Turntable](/assets/surfzar/surfzar-hero.gif)

---

## 2. Rendering Pure Implicit Zero-Sets $f(x, y, z) = 0$

Unlike conventional 3D engines that approximate curved surfaces with triangle meshes, SurfZar directly renders continuous zero-level sets of implicit polynomial and transcendental equations in $\mathbb{R}^3$:

$$\mathcal{S} = \{(x, y, z) \in \mathbb{R}^3 \mid f(x, y, z) = 0\}$$

For every pixel on the screen, a viewing ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ is cast through a bounding volume (Box, Sphere, Cylinder, or Torus). Substituting the ray equation into $f(x, y, z) = 0$ reduces the 3D surface intersection problem to finding the smallest positive real root $t > 0$ of a univariate function $g(t) = f(\mathbf{o} + t\mathbf{d}) = 0$.

![SurfZar Studio Rendering the Barth Sextic](/assets/surfzar/studio_barth.png)

---

## 3. The Hybrid Mathematical Engine: Analytical Solvers + Interval Raymarching

A single numerical method cannot render both low-degree smooth manifolds and high-degree singular surfaces without artifacts. SurfZar parses the user's mathematical expression into an Abstract Syntax Tree (`MathEngine.ts`), classifies its polynomial degree, and compiles specialized GLSL evaluation paths:

| Degree | Formulation | GPU Solver Strategy | Example Manifolds |
| :---: | :--- | :--- | :--- |
| **1–2** | Linear & Quadric forms | Exact quadratic formula | Sphere, Cylinder, Hyperboloid, Cone |
| **3** | $P_3(t) = c_3 t^3 + c_2 t^2 + c_1 t + c_0$ | **Cardano-Tartaglia** closed-form cubic solver | Clebsch Diagonal Cubic, Cayley Cubic, Ding-Dong |
| **4** | $P_4(t) = c_4 t^4 + c_3 t^3 + c_2 t^2 + c_1 t + c_0$ | **Ferrari** closed-form quartic resolvent | Torus, Steiner's Roman, Kummer, Cushion |
| **5–6** | Quintics & Sextics | Multi-interval root isolation + **Bisection (12 iters) & Newton-Raphson (8 iters)** | **Barth Sextic** (65 icosahedral singularities), Citric |
| **Transcendental** | $\sin, \cos, \tan, \exp, \ln, \sqrt[3]{x}$ | High-precision adaptive interval raymarcher with near-miss detection | Gyroid, Schwarz P, Calabi-Yau, Topological Waves |

### Solving Singularities and Thin-Sheet Artifacts

When rendering classic algebraic surfaces like **Steiner's Roman Surface** or the **Barth Sextic**, two notorious GPU shader artifacts appear:
1. **Parabolic Vertex Halo Noise**: Along self-intersecting axes where $\nabla f = \mathbf{0}$, standard sign-change detectors miss grazing roots or trigger false positives. SurfZar validates local extrema and near-miss intervals before discarding or refining a root.
2. **Two-Sided Manifold Orientation (`rayHitSign`)**: Implicit surfaces partition space into interior ($f(x,y,z) < 0$) and exterior ($f(x,y,z) > 0$). By tracking the sign of $f$ along the approaching ray prior to root refinement, SurfZar prevents normal-flipping shimmer on thin sheets and shades **Side A (Frontface)** and **Side B (Backface)** with independent PBR materials, normal maps, metallic/roughness parameters, and HDRI environment reflections.

![Barth Sextic 65 Singularities](/assets/surfzar/surfzar-barth.gif)

---

## 4. Parametric Timeline Animation & Watertight 3D Printing

Beyond static inspection, SurfZar is a complete creative studio for mathematical visualization:

- **Multi-Track Keyframe Dope-Sheet**: Animate parameters $a, b, c, d \in [-5, 5]$ alongside continuous time $T$ over a 0–10s timeline with linear, step, and smooth cubic interpolation, plus direct **WebM video capture** and **4K transparent PNG snapshots**.
- **Dynamic Cross-Section Clipping**: Slice through $\pm X, \pm Y, \pm Z$ axes in real time to inspect internal chambers inside minimal surfaces like the **Gyroid** (`sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0`).
- **Marching Cubes STL / OBJ Export**: With a single click, SurfZar polygonizes the implicit field into a watertight triangle mesh ready for 3D printing in Bambu Studio, PrusaSlicer, Cura, or Blender.
- **Lossless URL State Serialization**: Every equation, camera matrix, dual-sided PBR material, clipping plane, and keyframe track compresses into a shareable `#share=...` URL hash.

![Parametric Morphing Animation](/assets/surfzar/surfzar-morphing.gif)

---

## 5. Try SurfZar Live

- **SurfZar Website**: [https://surfzar.vercel.app/](https://surfzar.vercel.app/)
- **GitHub Repository**: [https://github.com/daniazar/SurfZar](https://github.com/daniazar/SurfZar)
