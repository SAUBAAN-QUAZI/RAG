'use client';
import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { ragApi } from '../api/ragApi';

/**
 * Home page component
 */
const Home: React.FC = () => {
  const [isConnected, setIsConnected] = useState<boolean | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Check API connection on mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        await ragApi.checkHealth();
        setIsConnected(true);
      } catch (error) {
        console.error('API connection error:', error);
        setIsConnected(false);
      } finally {
        setIsLoading(false);
      }
    };

    checkConnection();
  }, []);

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold mb-4">Welcome to the RAG System</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          A Retrieval-Augmented Generation (RAG) system that combines the power of large language
          models with document retrieval capabilities.
        </p>
      </div>

      {/* API Status */}
      <div className="mb-12 flex justify-center">
        <div className="bg-white rounded-md shadow-sm border p-4 inline-flex items-center">
          <div className="mr-3 text-gray-700">API Status:</div>
          {isLoading ? (
            <div className="text-gray-500">Checking connection...</div>
          ) : isConnected ? (
            <div className="flex items-center text-green-600">
              <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
              Connected
            </div>
          ) : (
            <div className="flex items-center text-red-600">
              <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
              Disconnected
            </div>
          )}
        </div>
      </div>

      {/* Feature Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
        <Link
          href="/upload"
          className="bg-white rounded-md shadow-sm border p-6 hover:shadow-md transition-shadow"
        >
          <h2 className="text-xl font-semibold mb-2">Upload Documents</h2>
          <p className="text-gray-600 mb-4">
            Upload PDF documents to the knowledge base for the RAG system to use when answering
            questions.
          </p>
          <div className="text-blue-600 font-medium">Upload &rarr;</div>
        </Link>

        <Link
          href="/chat"
          className="bg-white rounded-md shadow-sm border p-6 hover:shadow-md transition-shadow"
        >
          <h2 className="text-xl font-semibold mb-2">Chat with RAG</h2>
          <p className="text-gray-600 mb-4">
            Ask questions about your uploaded documents and get answers grounded in their content.
          </p>
          <div className="text-blue-600 font-medium">Chat &rarr;</div>
        </Link>

        <Link
          href="/settings"
          className="bg-white rounded-md shadow-sm border p-6 hover:shadow-md transition-shadow"
        >
          <h2 className="text-xl font-semibold mb-2">Settings</h2>
          <p className="text-gray-600 mb-4">
            Configure the RAG system behavior, including API settings and retrieval parameters.
          </p>
          <div className="text-blue-600 font-medium">Configure &rarr;</div>
        </Link>

        <a
          href="https://github.com/yourusername/RAG"
          target="_blank"
          rel="noopener noreferrer"
          className="bg-white rounded-md shadow-sm border p-6 hover:shadow-md transition-shadow"
        >
          <h2 className="text-xl font-semibold mb-2">GitHub Repository</h2>
          <p className="text-gray-600 mb-4">
            View the source code, contribute to the project, or report issues on GitHub.
          </p>
          <div className="text-blue-600 font-medium">View Code &rarr;</div>
        </a>
      </div>

      {/* How It Works */}
      <div className="bg-white rounded-md shadow-sm border p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">How It Works</h2>
        <ol className="list-decimal pl-5 space-y-3">
          <li className="text-gray-700">
            <span className="font-medium">Upload documents</span> - Add PDFs to the knowledge base
          </li>
          <li className="text-gray-700">
            <span className="font-medium">Backend processing</span> - Documents are split into
            chunks, embedded, and stored in a vector database
          </li>
          <li className="text-gray-700">
            <span className="font-medium">Ask questions</span> - Submit queries through the chat
            interface
          </li>
          <li className="text-gray-700">
            <span className="font-medium">Relevant retrieval</span> - The system finds the most
            relevant document chunks
          </li>
          <li className="text-gray-700">
            <span className="font-medium">AI-powered answers</span> - Get responses grounded in your
            documents
          </li>
        </ol>
      </div>
    </div>
  );
};

export default Home; 