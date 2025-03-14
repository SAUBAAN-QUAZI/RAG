'use client';
import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { ragApi } from '../api/ragApi';
import config from '../config';

/**
 * Home component for the RAG UI
 */
const Home: React.FC = () => {
    const [apiStatus, setApiStatus] = useState<'loading' | 'connected' | 'error'>('loading');
    const [errorMessage, setErrorMessage] = useState<string>('');
    const [isRetrying, setIsRetrying] = useState<boolean>(false);

    useEffect(() => {
        checkConnection();
    }, []);

    const checkConnection = async () => {
        try {
            setApiStatus('loading');
            setIsRetrying(false);
            setErrorMessage('');
            
            console.log(`Checking connection to API at: ${config.apiUrl}`);
            await ragApi.checkHealth();
            setApiStatus('connected');
        } catch (error) {
            console.error('API connection error:', error);
            setApiStatus('error');
            
            // Extract meaningful error message
            if (error instanceof Error) {
                setErrorMessage(error.message);
            } else {
                setErrorMessage('Failed to connect to the API server. Please check if the server is running.');
            }
        }
    };

    const handleRetry = () => {
        setIsRetrying(true);
        setTimeout(() => {
            checkConnection();
        }, 1000);
    };

    return (
        <div className="max-w-4xl mx-auto">
            <h1 className="text-3xl font-bold mb-6">Welcome to the RAG System</h1>
            
            {/* API Status */}
            <div className="mb-8 p-4 border rounded-lg bg-gray-50">
                <h2 className="text-xl font-semibold mb-2">System Status</h2>
                <div className="flex items-center mb-2">
                    <span className="mr-2">API Connection:</span>
                    {apiStatus === 'loading' && (
                        <span className="text-yellow-500 flex items-center">
                            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-yellow-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Checking...
                        </span>
                    )}
                    {apiStatus === 'connected' && (
                        <span className="text-green-500 flex items-center">
                            <svg className="h-4 w-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                            </svg>
                            Connected
                        </span>
                    )}
                    {apiStatus === 'error' && (
                        <span className="text-red-500 flex items-center">
                            <svg className="h-4 w-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                            </svg>
                            Disconnected
                        </span>
                    )}
                </div>
                
                <div className="text-sm text-gray-600 mb-2">
                    API URL: <code className="bg-gray-100 px-1 py-0.5 rounded">{config.apiUrl}</code>
                </div>

                {apiStatus === 'error' && (
                    <div className="mt-2">
                        <p className="text-red-600 text-sm mb-2">{errorMessage}</p>
                        <div className="bg-red-50 border-l-4 border-red-500 p-4 mt-2">
                            <div className="flex">
                                <div className="flex-shrink-0">
                                    <svg className="h-5 w-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                                    </svg>
                                </div>
                                <div className="ml-3">
                                    <p className="text-sm text-red-700">
                                        Please ensure that:
                                    </p>
                                    <ul className="mt-1 text-sm text-red-700 list-disc list-inside">
                                        <li>The API server is running</li>
                                        <li>The API URL is correct in your environment settings</li>
                                        <li>There are no network issues or firewall restrictions</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <button
                            onClick={handleRetry}
                            disabled={isRetrying}
                            className="mt-3 bg-blue-500 hover:bg-blue-600 text-white py-1 px-4 rounded-md text-sm flex items-center disabled:opacity-50"
                        >
                            {isRetrying ? (
                                <>
                                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Retrying...
                                </>
                            ) : (
                                'Retry Connection'
                            )}
                        </button>
                    </div>
                )}
            </div>

            {/* Quick Start Guide */}
            <div className="mb-6">
                <h2 className="text-2xl font-semibold mb-4">Get Started</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Link href="/upload" className="block p-6 bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors">
                        <h3 className="text-xl font-medium text-blue-800 mb-2">Upload Documents</h3>
                        <p className="text-blue-600">Add PDF documents to the RAG system for processing and querying.</p>
                    </Link>
                    <Link href="/chat" className="block p-6 bg-green-50 hover:bg-green-100 rounded-lg transition-colors">
                        <h3 className="text-xl font-medium text-green-800 mb-2">Chat with Documents</h3>
                        <p className="text-green-600">Ask questions about your uploaded documents and get AI-generated answers.</p>
                    </Link>
                </div>
            </div>

            {/* About */}
            <div className="mb-6">
                <h2 className="text-2xl font-semibold mb-4">About RAG</h2>
                <p className="mb-2">
                    The <strong>Retrieval-Augmented Generation (RAG)</strong> system combines the power of large language models with document 
                    retrieval capabilities to provide accurate and context-aware responses based on your documents.
                </p>
                <p>
                    Upload your PDFs, ask questions, and get answers grounded in the content of your documents.
                </p>
            </div>
        </div>
    );
};

export default Home; 