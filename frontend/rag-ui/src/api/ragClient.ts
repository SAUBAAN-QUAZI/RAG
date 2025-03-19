"use client";

/**
 * Client-side API wrapper
 * This file ensures API calls only happen on the client side
 * This file must have the "use client" directive to ensure it works in Next.js 13+
 */

import { ragApi } from './ragApi';
import type { 
  QueryRequest, 
  QueryResponse, 
  DocumentUploadResult,
  BatchUploadResultItem 
} from './ragApi';
import config from '../config';

// Re-export types for client-side usage
export type {
  QueryRequest,
  QueryResponse,
  DocumentUploadResult,
  BatchUploadResultItem
};

// Log configuration information on client-side import
console.log('Client API initialized with config:', {
  apiUrl: config.apiUrl,
  environment: process.env.NODE_ENV,
  version: config.version
});

// Export a client-safe version of the API
export const clientApi = {
  // API configuration for client components
  config: {
    apiUrl: config.apiUrl,
    version: config.version,
    isDevelopment: config.isDevelopment
  },
  
  // Query the RAG system
  query: async (queryRequest: QueryRequest): Promise<QueryResponse> => {
    return ragApi.query(queryRequest);
  },
  
  // Upload document
  uploadDocument: async (
    file: File,
    metadata?: { title?: string; author?: string; description?: string },
    onProgress?: (percentage: number) => void
  ): Promise<DocumentUploadResult> => {
    return ragApi.uploadDocument(file, metadata, onProgress);
  },
  
  // Upload multiple documents
  uploadMultipleDocuments: async (
    files: File[],
    metadata?: { titlePrefix?: string; author?: string; description?: string },
    onProgress?: (percentage: number) => void
  ): Promise<{ 
    message: string; 
    successful_count: number; 
    failed_count: number; 
    results: BatchUploadResultItem[] 
  }> => {
    return ragApi.uploadMultipleDocuments(files, metadata, onProgress);
  },
  
  // Check health
  checkHealth: async (): Promise<{ message: string }> => {
    return ragApi.checkHealth();
  },
  
  // List documents
  listDocuments: async (): Promise<Array<{ id: string; name: string; status: string; metadata: Record<string, any> }>> => {
    return ragApi.listDocuments();
  },
  
  // Delete document
  deleteDocument: async (documentId: string): Promise<{ status: string; message: string }> => {
    return ragApi.deleteDocument(documentId);
  }
}; 