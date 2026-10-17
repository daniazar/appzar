export interface ProjectLink {
  label: string;
  url: string;
  icon: 'code' | 'globe' | 'video' | 'play' | 'download';
}

export interface ProjectItem {
  id: string;
  title: string;
  year: string;
  tech: string[];
  description: string;
  note?: string;
  coAuthor?: string;
  image: string;
  accent: 'cyan' | 'purple' | 'amber' | 'indigo';
  links: ProjectLink[];
}

export const PROFILE = {
  name: 'Daniel Azar',
  role: 'Software & AI Systems Engineer',
  tagline: 'AI Systems & Autonomous Agents • GPU Inference • Computer Graphics • Cloud Architecture',
  heroSummary:
    '15+ years architecting full-stack cloud platforms, interactive 3D graphics, agentic AI systems, and sub-100ms GPU neural pipelines — across high-paced AI startups like Scale AI, enterprise cloud teams, and my own ventures.',
  location: 'Argentina (Available Worldwide)',
  bio: [
    'I am a Software & AI Systems Engineer with 15+ years of experience designing and shipping production software across AI, Computer Graphics, Cloud Architecture, and Full-Stack Web. Based in Argentina and collaborating worldwide, I have worked at high-paced AI startups like Scale AI, served as a Technical Lead & Principal Engineer, and co-founded the cloud & SecDevOps consultancy MoltenMinds.',
    'I hold an MBA (Master in Business Administration, 2025) where my thesis researched autonomous AI business analysts and recursive Fibonacci agile backlog engines (Molten AI Agent), alongside a Software Engineering degree (Ingeniería Informática) from ITBA (2013), where my undergraduate thesis pioneered real-time per-pixel GPU algebraic surface ray tracing on mobile silicon (iSurfer).',
    'My recent engineering work spans autonomous multi-agent workflows (CineZero, Molten AI Agent, Jules), real-time GPU inference on NVIDIA Blackwell (TensorRT FP8/FP16, CUDA Graphs, fused Triton kernels, zero-copy NVENC/WebRTC streaming, and neural TTS), and browser-native 3D mathematical and spatial studios (SurfZar, ArchitecturaAI) built with WebGL2, WebGPU, Next.js, and TypeScript.',
  ],
  highlights: [
    { value: '15+ yrs', label: 'Production Software & Systems Engineering' },
    { value: 'Scale AI', label: 'High-Paced AI Startup & Tech Lead Experience' },
    { value: '90+ FPS', label: 'Blackwell FP8/FP16 Real-Time GPU Video AI' },
    { value: 'MBA + ITBA', label: 'Business Administration & Software Engineering' },
  ],
  education: [
    {
      degree: 'MBA — Master in Business Administration',
      institution: 'Master’s Degree',
      year: '2025',
      thesis: 'Thesis: Molten AI Agent — Autonomous Business Analyst & Recursive Fibonacci Backlog Engine',
      thesisUrl: '/posts/molten-ai-agent-master-thesis',
    },
    {
      degree: 'Software Engineering (Ingeniería Informática)',
      institution: 'ITBA — Instituto Tecnológico de Buenos Aires',
      year: '2013',
      thesis: 'Thesis: iSurfer — Real-Time GPU Algebraic Surface Ray Tracer in OpenGL ES 2.0',
      thesisUrl: '/posts/isurfer-gpu-algebraic-surface-raytracer',
    },
  ],
  languages: [
    { name: 'Spanish', level: 'Native' },
    { name: 'English', level: 'Fluent / Bilingual Professional' },
  ],
  skills: [
    {
      category: 'AI, GPU & Computer Graphics',
      items: [
        'PyTorch',
        'CUDA Graphs',
        'TensorRT (FP8/FP16)',
        'Triton Kernels',
        'Agentic AI & MCP',
        'WebGPU & WebGL2',
        'ONNX Runtime',
        'GLSL Shaders',
        'NVENC / NVDEC',
        'Three.js / R3F',
      ],
    },
    {
      category: 'Backend, Streaming & Cloud',
      items: ['TypeScript / Node.js', 'Python / FastAPI', 'C# / .NET', 'Go', 'Java / Spring', 'C++ SIMD AVX2', 'WebRTC', 'Supabase / PostgreSQL', 'Azure / AWS / GCP', 'SecDevOps'],
    },
    {
      category: 'Frontend, Desktop & Mobile',
      items: ['React 19', 'Next.js 15/16', 'Tailwind CSS', 'Angular', 'Remotion', 'Dart / Flutter', 'Electron', 'WinUI 3'],
    },
  ],
  socials: [
    {
      label: 'GitHub',
      url: 'https://github.com/daniazar',
      icon: 'github',
      hoverColor: 'hover:text-white hover:border-white/40',
    },
    {
      label: 'LinkedIn',
      url: 'https://ar.linkedin.com/in/daniazar/',
      icon: 'linkedin',
      hoverColor: 'hover:text-[#0a66c2] hover:border-[#0a66c2]/50',
    },
    {
      label: 'Twitter / X',
      url: 'https://twitter.com/azardani',
      icon: 'twitter',
      hoverColor: 'hover:text-[#4099ff] hover:border-[#4099ff]/50',
    },
    {
      label: 'Instagram',
      url: 'https://www.instagram.com/daniazar.sh/',
      icon: 'instagram',
      hoverColor: 'hover:text-[#e1306c] hover:border-[#e1306c]/50',
    },
    {
      label: 'Facebook',
      url: 'https://www.facebook.com/daniel.j.azar',
      icon: 'facebook',
      hoverColor: 'hover:text-[#3b5998] hover:border-[#3b5998]/50',
    },
    {
      label: 'WhatsApp',
      url: 'https://wa.me/+5491155795814',
      icon: 'whatsapp',
      hoverColor: 'hover:text-[#25d366] hover:border-[#25d366]/50',
    },
  ],
  email: {
    name: 'danielazar',
    domain: 'gmail',
    tld: 'com',
  },
};

export const PROJECTS: ProjectItem[] = [
  {
    id: 'ruby-newsstudio',
    title: 'Ruby NewsStudio — Real-Time AI Broadcast Engine',
    year: '2026',
    tech: ['TensorRT 10.15', 'CUDA Graphs', 'Triton NV12', 'PyTorch', 'WebRTC', 'WebGPU', 'Next.js 15'],
    description:
      'An autonomous AI broadcast studio and full-duplex conversational video anchor engine. Combines Blackwell FP8/FP16 TensorRT talking-head pipelines (Ditto & MuseTalk at 50–90+ FPS), RIFE optical-flow frame doubling, fused Triton NV12 splicing, and sub-100ms multi-engine neural TTS.',
    image: '/images/thumbs/project_ruby_newsstudio.webp',
    accent: 'cyan',
    links: [
      {
        label: 'Read 36-Part Engineering Series',
        url: '#blog',
        icon: 'code',
      },
    ],
  },
  {
    id: 'surfzar',
    title: 'SurfZar — Real-Time 3D Algebraic Surface Studio',
    year: '2026',
    tech: ['WebGL2 / GLSL', 'Three.js / R3F', 'Next.js 16', 'React 19', 'TypeScript', 'Marching Cubes STL'],
    description:
      'The modern browser evolution of my iSurfer thesis: a zero-install 3D mathematical studio that renders implicit algebraic and transcendental surfaces f(x,y,z)=0 at 60 FPS using closed-form Cardano/Ferrari root solvers, interval raymarching, dual-sided PBR shading, keyframe timelines, and watertight 3D print STL export.',
    image: '/images/thumbs/project_surfzar.webp',
    accent: 'purple',
    links: [
      {
        label: 'Open Website',
        url: 'https://surfzar.vercel.app/',
        icon: 'globe',
      },
      {
        label: 'View Code',
        url: 'https://github.com/daniazar/SurfZar',
        icon: 'code',
      },
      {
        label: 'Architecture',
        url: '/posts/surfzar_realtime_3d_algebraic_surfaces_webgl',
        icon: 'play',
      },
      {
        label: 'Curvature & CSG',
        url: '/posts/surfzar-differential-geometry-csg-and-3d-printing',
        icon: 'play',
      },
      {
        label: 'GF(2⁸) QR Engine',
        url: '/posts/surfzar-reed-solomon-qr-encoder-and-url-state',
        icon: 'play',
      },
    ],
  },
  {
    id: 'cinezero',
    title: 'CineZero — Autonomous AI Filmmaking Studio',
    year: '2026',
    tech: ['Next.js', 'TypeScript', 'Veo 3.1', 'Google Flow', 'FFmpeg', 'Gemini API'],
    description:
      'An assistant-driven AI filmmaking studio that takes an idea through Story, World, Script, Produce, and Cut. Features multi-angle character reference sheets, sharpest-frame seam chaining across 8-second Veo clips, mandatory cost ceilings, and content-addressed segment reuse.',
    image: '/images/thumbs/cinezero-autonomous-ai-filmmaking-studio.webp',
    accent: 'amber',
    links: [
      {
        label: 'View Code',
        url: 'https://github.com/daniazar/CineZero',
        icon: 'code',
      },
      {
        label: 'Read Article',
        url: '/posts/cinezero-autonomous-ai-filmmaking-studio',
        icon: 'play',
      },
    ],
  },
  {
    id: 'google-flow-mcp',
    title: 'Google Flow MCP — 36-Tool Veo 3.1 & Imagen 4 Server',
    year: '2026',
    tech: ['TypeScript', 'Model Context Protocol', 'tRPC', 'Playwright', 'FFmpeg', 'Vitest'],
    description:
      'A two-tier Model Context Protocol (MCP) server for Google Labs Flow that pairs direct authenticated HTTP/tRPC API execution with persistent Playwright fallback, media-ID frame verification, and a local credit spend ledger.',
    image: '/images/thumbs/google-flow-mcp-trpc-video-server.webp',
    accent: 'cyan',
    links: [
      {
        label: 'View Code',
        url: 'https://github.com/daniazar/google-flow-mcp',
        icon: 'code',
      },
      {
        label: 'Read Article',
        url: '/posts/google-flow-mcp-trpc-video-server',
        icon: 'play',
      },
    ],
  },
  {
    id: 'antigravity-jules-plugin',
    title: 'Antigravity Jules Plugin — Cloud Agent Delegation',
    year: '2026',
    tech: ['Google Antigravity', 'Google Jules CLI', 'AI Agents', 'Git Automation'],
    description:
      'An open-source plugin for Google Antigravity IDE that delegates parallel coding tasks, bug fixes, and test suites to asynchronous Google Jules cloud VMs with non-destructive diff inspection and dirty-worktree guards.',
    image: '/images/thumbs/antigravity-jules-plugin.webp',
    accent: 'purple',
    links: [
      {
        label: 'View Code',
        url: 'https://github.com/daniazar/antigravity-jules-plugin',
        icon: 'code',
      },
      {
        label: 'Read Article',
        url: '/posts/antigravity-jules-plugin',
        icon: 'play',
      },
    ],
  },
  {
    id: 'propertyspain',
    title: 'VallecasCasa (PropertySpain) — 30-Year Rent-to-Own PropTech',
    year: '2026',
    tech: ['Next.js 14', 'TypeScript', 'Supabase RLS', 'next-intl', 'Framer Motion', 'Tailwind CSS'],
    description:
      'A bilingual Madrid Rent-to-Own (Alquiler con Opción a Compra) real estate platform featuring an interactive 30-year equity simulator, notarial clause inspector, progressive 2-step viability funnel, and admin CRM.',
    image: '/images/thumbs/propertyspain-vallecascasa-rent-to-own-proptech.webp',
    accent: 'cyan',
    links: [
      {
        label: 'View Code',
        url: 'https://github.com/daniazar/propertyspain',
        icon: 'code',
      },
      {
        label: 'Read Article',
        url: '/posts/propertyspain-vallecascasa-rent-to-own-proptech',
        icon: 'play',
      },
    ],
  },
  {
    id: 'architectura-ai',
    title: 'ArchitecturaAI — Virtual Staging & Renovation Studio',
    year: '2026',
    tech: ['Next.js 16', 'React 19', 'Gemini 2.5 Flash', 'Three.js / R3F', 'ReactFlow', 'Supabase', 'Stripe'],
    description:
      'An AI virtual staging and architectural renovation studio with structural geometry locks, non-destructive ReactFlow variation DAGs, 3D floor-plan viewers, and Stripe credit billing.',
    image: '/images/thumbs/architectura-ai-virtual-renovation-studio.webp',
    accent: 'indigo',
    links: [
      {
        label: 'View Code',
        url: 'https://github.com/daniazar/real-estate-renovation-studio',
        icon: 'code',
      },
      {
        label: 'Studio Overview',
        url: '/posts/architectura-ai-virtual-renovation-studio',
        icon: 'play',
      },
      {
        label: 'Image Tree DAG',
        url: '/posts/architectura-ai-image-tree-dag-versioning',
        icon: 'play',
      },
    ],
  },
  {
    id: 'pdfmaster',
    title: 'PdfMaster — Zero-Upload Client-Side PDF Editor',
    year: '2026',
    tech: ['Next.js 15', 'TypeScript', 'pdf.js', 'pdf-lib', 'Zustand', 'Tailwind CSS'],
    description:
      'A privacy-first cross-platform PDF editor that executes page reordering, rotation, merging, vector annotations, and full-text Ctrl+F search 100% client-side in the browser without uploading files.',
    image: '/images/thumbs/pdfmaster-cross-platform-pdf-editor.webp',
    accent: 'purple',
    links: [
      {
        label: 'View Code',
        url: 'https://github.com/daniazar/PdfMaster',
        icon: 'code',
      },
      {
        label: 'Read Article',
        url: '/posts/pdfmaster-cross-platform-pdf-editor',
        icon: 'play',
      },
    ],
  },
  {
    id: 'contract-app',
    title: 'ContractApp — AI Legal Contract Wizard & Clause Engine',
    year: '2025',
    tech: ['Next.js', 'TypeScript', 'Google Genkit AI', 'Material UI', 'Playwright'],
    description:
      'A modular legal contract lifecycle suite combining an AI-assisted questionnaire, drag-and-drop clause library, real-time document preview, and digital signature stamping.',
    image: '/images/thumbs/contractapp-ai-legal-contract-wizard.webp',
    accent: 'indigo',
    links: [
      {
        label: 'View Code',
        url: 'https://github.com/daniazar/contracts-app',
        icon: 'code',
      },
      {
        label: 'Read Article',
        url: '/posts/contractapp-ai-legal-contract-wizard',
        icon: 'play',
      },
    ],
  },
  {
    id: 'molten-ai-agent',
    title: 'Molten AI Agent — Autonomous Business Analyst & Agile Backlog Engine',
    year: '2025',
    tech: ['Next.js 14', 'TypeScript', 'Vercel AI SDK', 'OpenAI', 'Zod Schemas', 'Supabase RLS', 'Tailwind CSS'],
    description:
      "My MBA Master's Thesis project: a 7-stage agentic AI platform that transforms raw product ideas into structured Business Analyst specifications, generates role-based tasks with Fibonacci story points, recursively decomposes complex tasks (≥8 pts) into Epics and sub-tasks, and manages execution on a persistent Kanban board.",
    image: '/images/thumbs/molten-ai-agent-master-thesis.webp',
    accent: 'amber',
    links: [
      {
        label: 'View Code',
        url: 'https://github.com/daniazar/molten-ai-agent',
        icon: 'code',
      },
      {
        label: 'Read Thesis Post',
        url: '/posts/molten-ai-agent-master-thesis',
        icon: 'play',
      },
      {
        label: 'Fibonacci Engine',
        url: '/posts/molten-ai-recursive-fibonacci-task-decomposition',
        icon: 'play',
      },
    ],
  },
  {
    id: 'discord-schedule-bot',
    title: 'Discord Schedule Bot & Realtime UTC Dashboard',
    year: '2025',
    tech: ['Discord.js v14', 'Next.js', 'Supabase Realtime', 'TypeScript', 'WebSockets'],
    description:
      'A dual-interface community event scheduler pairing Discord slash commands (/add, /title) with a live WebSocket-synchronized Next.js dashboard normalized to UTC hours.',
    image: '/images/thumbs/discord-schedule-bot-realtime-dashboard.webp',
    accent: 'purple',
    links: [
      {
        label: 'View Code',
        url: 'https://github.com/daniazar/discord-schedule-bot',
        icon: 'code',
      },
      {
        label: 'Read Article',
        url: '/posts/discord-schedule-bot-realtime-dashboard',
        icon: 'play',
      },
    ],
  },
  {
    id: 'property-manager',
    title: 'Property Manager — Offline-First Rental Portfolio Suite',
    year: '2025',
    tech: ['Next.js', 'TypeScript', 'IndexedDB', 'Recharts', 'Radix UI', 'Google Drive API'],
    description:
      'A local-first real estate portfolio manager powered by a 10-store IndexedDB engine (propManagerDB) that tracks rental properties, multi-step lease escalations, tenant payments, automated tax projections, and encrypted Google Drive backups.',
    image: '/images/thumbs/property-manager-offline-indexeddb-rental-suite.webp',
    accent: 'cyan',
    links: [
      {
        label: 'View Code',
        url: 'https://github.com/daniazar/property-manager',
        icon: 'code',
      },
      {
        label: 'Read Article',
        url: '/posts/property-manager-offline-indexeddb-rental-suite',
        icon: 'play',
      },
    ],
  },
  {
    id: 'aoe2-de-ui-mod',
    title: 'Age of Empires II: DE — 4K UltraWide HUD Mod',
    year: '2020',
    tech: ['Age of Empires II: DE', 'JSON WidgetUI', 'DDS Textures', '4K UltraWide UX'],
    description:
      'A custom competitive 4K UltraWide UI mod (#16939 on the official Age of Empires Mod Workshop) that clusters the minimap, resource bar, unit queues, and command panel into the lower-left focal zone with transparent overlays.',
    image: '/images/thumbs/project_aoe2_ui_mod.webp',
    accent: 'amber',
    links: [
      {
        label: 'AoE2 Mod Workshop',
        url: 'https://www.ageofempires.com/mods/details/16939',
        icon: 'globe',
      },
      {
        label: 'View Code',
        url: 'https://github.com/daniazar/dani-ui-age2de',
        icon: 'code',
      },
      {
        label: 'Read Article',
        url: '/posts/aoe2-de-4k-ultrawide-ui-mod',
        icon: 'play',
      },
    ],
  },
  {
    id: 'lsystemts-turtle-ts',
    title: 'LSystemTS & Turtle-TS — Generative Fractal Engine',
    year: '2020',
    tech: ['TypeScript', 'HTML5 Canvas', 'L-Systems', 'Fractals', 'npm Package'],
    description:
      'A pair of open-source TypeScript libraries: a 2D vector Turtle Graphics engine (turtle-ts) and a recursive Lindenmayer System grammar evaluator (lsystemts) for rendering Sierpinski Triangles, Koch Snowflakes, and Hilbert space-filling curves.',
    image: '/images/thumbs/lsystemts-fractals-and-turtle-graphics.webp',
    accent: 'cyan',
    links: [
      {
        label: 'LSystemTS Repo',
        url: 'https://github.com/daniazar/lsystemts',
        icon: 'code',
      },
      {
        label: 'Turtle-TS Repo',
        url: 'https://github.com/daniazar/turtle-ts',
        icon: 'code',
      },
      {
        label: 'Read Article',
        url: '/posts/lsystemts-fractals-and-turtle-graphics',
        icon: 'play',
      },
    ],
  },
  {
    id: 'image-metadata-editor',
    title: 'Image Metadata Editor',
    year: '2020',
    tech: ['C#', 'WinUI 3 Alpha', 'Bing Maps API', 'Windows 10'],
    description:
      'A desktop application to batch-edit EXIF image metadata (dates and geolocation coordinates). Built during the COVID-19 quarantine to digitize and organize thousands of scanned family photo albums.',
    image: '/images/thumbs/project_image_metadata.webp',
    accent: 'purple',
    links: [
      {
        label: 'View Code',
        url: 'https://github.com/daniazar/ImageMetadataEditor',
        icon: 'code',
      },
      {
        label: 'Read Article',
        url: '/posts/fixing-old-images',
        icon: 'play',
      },
    ],
  },
  {
    id: 'tv-master',
    title: 'TV Master',
    year: '2018',
    tech: ['TypeScript', 'Angular', 'Firebase', 'Electron', 'Cordova'],
    description:
      'A responsive web and desktop application to watch Latin American television channels online for free with an ad-free, theater-mode interface, multi-source switching, and live event schedules.',
    note: 'Works best with Opera web browser, Chrome on Android, or the Electron desktop client.',
    image: '/images/thumbs/project_tvmaster.webp',
    accent: 'indigo',
    links: [
      {
        label: 'Open Website',
        url: 'https://tv.appzar.com.ar',
        icon: 'globe',
      },
      {
        label: 'View Code',
        url: 'https://github.com/daniazar/TVMaster',
        icon: 'code',
      },
      {
        label: 'Read Article',
        url: '/posts/making-an-app-to-watch-tv-channels',
        icon: 'play',
      },
    ],
  },
  {
    id: 'cloudcam',
    title: 'CloudCam',
    year: '2015',
    tech: ['JavaScript', 'AngularJS', 'PostgreSQL', 'Kairos', 'Raspberry Pi'],
    description:
      'A smart security camera platform with real-time face recognition. Supports standard webcams, most IP cameras, and a custom Raspberry Pi hardware camera prototype.',
    image: '/images/thumbs/project_cloudcam.webp',
    accent: 'amber',
    links: [
      {
        label: 'Watch Demo on YouTube',
        url: 'https://youtu.be/tSpp2GaxFvQ',
        icon: 'video',
      },
      {
        label: 'Read Article',
        url: '/posts/cloudcam-raspberry-pi-face-recognition',
        icon: 'play',
      },
    ],
  },
  {
    id: 'isurfer',
    title: 'iSurfer — Real-Time GPU Algebraic Surface Ray Tracer',
    year: '2013',
    tech: ['OpenGL ES 2.0', 'GLSL Shaders', 'Objective-C', 'AST Compiler', 'Ray Tracing'],
    description:
      'My university thesis project at ITBA: a real-time algebraic surfaces renderer for iPhone and iPad. Compiles user equations f(x,y,z)=0 on the fly via a recursive-descent AST parser into dynamically linked OpenGL ES 2.0 fragment shaders for per-pixel GPU ray tracing.',
    image: '/images/thumbs/project_isurfer.webp',
    accent: 'cyan',
    links: [
      {
        label: 'Watch Video',
        url: 'https://youtu.be/Y3HUX5ARSs8',
        icon: 'video',
      },
      {
        label: 'Part 1: Ray Tracing vs. Meshes',
        url: '/posts/isurfer-part-1-why-raytracing-beats-triangle-meshes-for-algebraic-surfaces',
        icon: 'play',
      },
      {
        label: 'Part 2: AST & GLSL JIT Compiler',
        url: '/posts/isurfer-part-2-equation-parser-ast-and-on-the-fly-glsl-compiler',
        icon: 'code',
      },
      {
        label: 'Part 3: Fragment Shader Ray Tracer',
        url: '/posts/isurfer-gpu-raytracing-algebraic-surfaces-ios',
        icon: 'play',
      },
    ],
  },
  {
    id: 'dream-list',
    title: 'Dream List',
    year: '2012',
    tech: ['C#', 'JavaScript', 'Cordova', 'BlackBerry World'],
    description:
      'A mobile dream journal and tracking application designed and published for the BlackBerry World Store.',
    image: '/images/thumbs/project_dreamlist.webp',
    accent: 'indigo',
    links: [
      {
        label: 'Store Archive',
        url: 'https://appworld.blackberry.com/webstore/content/35607888/?lang=en&countrycode=US',
        icon: 'download',
      },
    ],
  },
  {
    id: 'crypto-audio-steganography',
    title: 'Crypto Audio Steganography',
    year: '2011',
    tech: ['C', 'AES / DES / SHA', 'WAV Audio'],
    description:
      'A low-level cryptography and steganography tool in C to encrypt information using AES, DES, and SHA and conceal the payload inside uncompressed WAV audio files.',
    coAuthor: 'Made with Gustavo Maldonado',
    image: '/images/thumbs/project_stenography.webp',
    accent: 'amber',
    links: [
      {
        label: 'View Code',
        url: 'https://github.com/daniazar/cripto2011',
        icon: 'code',
      },
      {
        label: 'Read Article',
        url: '/posts/crypto-audio-steganography-wav-c',
        icon: 'play',
      },
    ],
  },
  {
    id: 'asteroids-deluxe',
    title: 'Asteroids Deluxe',
    year: '2010',
    tech: ['C#', 'Unity 3D', 'Game Dev'],
    description:
      'An Asteroids game built for a university course featuring two playable camera modes: 2D Classic and 2.5D third-person chase camera behind the spacecraft.',
    image: '/images/thumbs/project_asteroids.webp',
    accent: 'purple',
    links: [
      {
        label: 'Play Game',
        url: 'https://daniazar.github.io/ITBA-asteroids/',
        icon: 'play',
      },
      {
        label: 'View Code',
        url: 'https://github.com/daniazar/ITBA-asteroids',
        icon: 'code',
      },
      {
        label: 'Read Article',
        url: '/posts/unity-procedural-city-racing-and-asteroids',
        icon: 'code',
      },
    ],
  },
  {
    id: 'city-racing',
    title: 'City Racing',
    year: '2010',
    tech: ['C#', 'Unity 3D', 'Procedural Generation', 'AI'],
    description:
      'A procedurally generated racing game with infinite city courses, dynamic road generation, and AI opponents. Published on Kongregate where over 10,000 players raced it.',
    note: 'Legacy Unity Web Player build on Kongregate.',
    image: '/images/thumbs/project_cityracer.webp',
    accent: 'indigo',
    links: [
      {
        label: 'Kongregate Page',
        url: 'https://www.kongregate.com/games/daniazar/city-racing',
        icon: 'play',
      },
      {
        label: 'View Code',
        url: 'https://github.com/daniazar/ITBA-CityRace',
        icon: 'code',
      },
      {
        label: 'Read Article',
        url: '/posts/unity-procedural-city-racing-and-asteroids',
        icon: 'code',
      },
    ],
  },
];
