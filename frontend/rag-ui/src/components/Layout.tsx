import React, { ReactNode } from 'react';
import Link from 'next/link';

interface LayoutProps {
  children: ReactNode;
}

/**
 * Main layout component for the RAG UI
 */
const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Header */}
      <header className="bg-blue-600 text-white shadow-md">
        <div className="container mx-auto px-4 py-3">
          <div className="flex justify-between items-center">
            <div className="text-xl font-bold">RAG System</div>
            <nav>
              <ul className="flex space-x-6">
                <li>
                  <Link href="/" className="hover:underline">
                    Home
                  </Link>
                </li>
                <li>
                  <Link href="/upload" className="hover:underline">
                    Upload
                  </Link>
                </li>
                <li>
                  <Link href="/chat" className="hover:underline">
                    Chat
                  </Link>
                </li>
                <li>
                  <Link href="/settings" className="hover:underline">
                    Settings
                  </Link>
                </li>
              </ul>
            </nav>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-grow container mx-auto px-4 py-8">
        {children}
      </main>

      {/* Footer */}
      <footer className="bg-gray-100 border-t">
        <div className="container mx-auto px-4 py-4">
          <div className="text-center text-gray-600 text-sm">
            RAG System &copy; {new Date().getFullYear()} - Powered by Next.js and FastAPI
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Layout; 