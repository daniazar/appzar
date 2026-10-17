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

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}
