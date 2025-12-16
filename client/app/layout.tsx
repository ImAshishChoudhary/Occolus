import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Occolus',
  description: 'AI-powered drug discovery and research platform',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><path d='M16 2L29 10v12L16 30 3 22V10z' fill='%231a1a1a'/><circle cx='16' cy='10' r='2.5' fill='%23f5f0e8'/><circle cx='10' cy='20' r='2.5' fill='%23f5f0e8'/><circle cx='22' cy='20' r='2.5' fill='%23f5f0e8'/><path d='M16 10L10 20M16 10L22 20M10 20H22' stroke='%23f5f0e8' stroke-width='1.5'/></svg>" type="image/svg+xml" />
      </head>
      <body className={inter.className + ' font-space-grotesk'}>{children}</body>
    </html>
  );
}
