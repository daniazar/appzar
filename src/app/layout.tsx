import type { Metadata } from 'next';
import 'katex/dist/katex.min.css';
import './globals.css';

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || 'https://www.appzar.com.ar';

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: 'Daniel Azar — Software Engineer • AI, Graphics & Cloud Blog (AppZar)',
    template: '%s — Daniel Azar (AppZar)',
  },
  description:
    'Personal portfolio, real-time neural video & audio acceleration research, computer graphics experiments, and engineering blog by Daniel Azar.',
  alternates: {
    canonical: '/',
    types: {
      'application/rss+xml': `${SITE_URL}/rss.xml`,
    },
  },
  openGraph: {
    type: 'website',
    url: SITE_URL,
    siteName: 'Daniel Azar — AppZar Engineering Blog',
    title: 'Daniel Azar — Software Engineer • AI, Graphics & Cloud Blog (AppZar)',
    description:
      'Deep-dive engineering articles on CUDA Graphs, TensorRT FP8/FP16, neural talking-head avatars, WebGL2 algebraic surfaces, and cloud architecture.',
    images: [
      {
        url: '/images/thumbs/project_ruby_newsstudio.webp',
        width: 1200,
        height: 630,
        alt: 'Daniel Azar — AppZar Engineering Portfolio & Blog',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    creator: '@azardani',
    title: 'Daniel Azar — Software Engineer • AI, Graphics & Cloud Blog (AppZar)',
    description:
      'Deep-dive engineering articles on CUDA Graphs, TensorRT FP8/FP16, neural talking-head avatars, WebGL2 algebraic surfaces, and cloud architecture.',
    images: ['/images/thumbs/project_ruby_newsstudio.webp'],
  },
  icons: {
    icon: '/favicon.ico',
    apple: '/apple-touch-icon.png',
  },
};

const personJsonLd = {
  '@context': 'https://schema.org',
  '@type': 'ProfilePage',
  url: SITE_URL,
  mainEntity: {
    '@type': 'Person',
    name: 'Daniel Azar',
    url: SITE_URL,
    jobTitle: 'Software AI Engineer',
    worksFor: {
      '@type': 'Organization',
      name: 'MoltenMinds',
      url: 'https://www.moltenminds.com',
    },
    description:
      'Software AI Engineer with 15+ years building multimodal AI pipelines, 3D sensor calibration & computer graphics engines, and cloud-scale distributed systems at Scale AI, Microsoft Azure (via SOUTHWORKS), MoltenMinds, and Globant.',
    alumniOf: [
      {
        '@type': 'CollegeOrUniversity',
        name: 'Universidad de Palermo (MBA)',
      },
      {
        '@type': 'CollegeOrUniversity',
        name: 'ITBA — Instituto Tecnológico de Buenos Aires',
      },
      {
        '@type': 'EducationalOrganization',
        name: 'Escuela Técnica ORT',
      },
    ],
    knowsAbout: [
      'Artificial Intelligence',
      'Multimodal & Generative AI',
      '3D Sensor Fusion & Computer Graphics',
      'CUDA Graphs & TensorRT',
      'WebGL2 & WebGPU',
      'Microsoft Azure & Cloud Architecture',
      'TypeScript & Next.js',
    ],
    sameAs: [
      'https://github.com/daniazar',
      'https://ar.linkedin.com/in/daniazar/',
      'https://twitter.com/azardani',
      'https://www.instagram.com/daniazar.sh/',
    ],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(personJsonLd) }}
        />
        {children}
      </body>
    </html>
  );
}
