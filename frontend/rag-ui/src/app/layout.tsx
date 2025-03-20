import type { Metadata } from "next";
import "./globals.css";
import { Providers } from './providers';
import dynamic from 'next/dynamic';

const ConfigCheck = dynamic(() => import('../components/ConfigCheck'), { ssr: false });

export const metadata: Metadata = {
  title: "RAG UI - Document Q&A System",
  description: "A modern interface for Retrieval-Augmented Generation",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Providers>
          {/* Runtime config check - will show warning if API URL is wrong */}
          <ConfigCheck />
          {children}
        </Providers>
      </body>
    </html>
  );
}
