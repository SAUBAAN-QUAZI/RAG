import axios, { AxiosError, AxiosInstance } from 'axios';
import config from '../config';

/**
 * Helper to safely check if we're in a browser environment
 * This pattern is reliable for Next.js's React Server Components
 */
const isBrowser = (): boolean => {
  return typeof window !== 'undefined' && typeof window.document !== 'undefined';
};

// Network health check state
let isNetworkIssueReported = false;

// Placeholder for the API client
let api: AxiosInstance | null = null;

// Initialize the API client (safe for SSR)
const initializeApi = (): void => {
  // Only initialize in browser environments
  if (!isBrowser()) return;

  // Ensure API URL doesn't have trailing slash
  let baseUrl = config.apiUrl;
  if (baseUrl.endsWith('/')) {
    baseUrl = baseUrl.slice(0, -1);
  }

  console.log('Initializing API client with base URL:', baseUrl);

  // Create an axios instance with default config
  api = axios.create({
    baseURL: baseUrl,
    headers: {
      'Content-Type': 'application/json',
    },
    // Add timeout to prevent hanging requests
    timeout: config.baseTimeout || 300000, // Use configured timeout or 5 minutes default
    // Add better retry behavior
    validateStatus: status => status < 500, // Treat 500+ as errors
  });

  // Log the API configuration
  console.log('API Client Configuration:', {
    baseURL: api.defaults.baseURL,
    environment: config.environment,
    version: config.version,
    isProduction: config.isProduction
  });

  // Safety check - we should never use localhost in production
  if (config.isProduction && api.defaults.baseURL?.includes('localhost')) {
    console.error('CRITICAL ERROR: Using localhost in production environment!');
    console.error('This would cause requests to fail or go to the wrong endpoint.');
    // Force override to production URL
    api.defaults.baseURL = 'https://rag-bpql.onrender.com';
    console.log('Forced API URL to production backend:', api.defaults.baseURL);
  }

  // Fix URL paths for API calls
  const resolveApiPath = (path: string): string => {
    // First, ensure the path starts with a slash if not empty
    if (path && !path.startsWith('/')) {
      path = '/' + path;
    }
    
    // In development, we need to map to the proper endpoints to match the backend
    // The backend has a mix of /api and non-/api endpoints
    const endpointsWithoutApiPrefix = ['/documents', '/documents/batch', '/health'];
    
    if (endpointsWithoutApiPrefix.includes(path)) {
      // These endpoints don't need /api prefix
      return path;
    }
    
    // For other endpoints that don't already have /api prefix, add it
    if (!path.startsWith('/api/')) {
      path = '/api' + path;
    }
    
    return path;
  };

  // Add request interceptor for logging and path fixing
  api.interceptors.request.use(request => {
    // Fix URL paths to ensure they're properly formatted
    if (request.url) {
      request.url = resolveApiPath(request.url);
    }

    // Remove duplicate /api prefixes if any
    if (request.url && request.url.includes('/api/api/')) {
      request.url = request.url.replace('/api/api/', '/api/');
      console.log('Fixed duplicate /api prefix in URL path');
    }

    // Add version info to query parameters for cache busting
    if (request.params) {
      request.params.version = config.version;
    } else {
      request.params = { version: config.version };
    }

    console.log(`API Request: ${request.method?.toUpperCase()} ${request.baseURL}${request.url}`, {
      params: request.params,
      environment: config.environment
    });
  
    // Reset network issue flag for new requests
    if (request.url !== '/health' && request.url !== '/api/health') {
      isNetworkIssueReported = false;
    }
  
    return request;
  });

  // Add response interceptor for logging
  api.interceptors.response.use(
    response => {
      console.log(`API Response: ${response.status} ${response.statusText}`);
      return response;
    },
    error => {
      // Log the error details
      if (axios.isAxiosError(error)) {
        const axiosError = error as AxiosError;
        
        // Log the basic error info
        console.error('API Error:', {
          message: axiosError.message,
          code: axiosError.code,
          status: axiosError.response?.status,
          statusText: axiosError.response?.statusText,
          url: axiosError.config?.url,
          baseURL: axiosError.config?.baseURL,
          environment: config.environment
        });
        
        // Check for backend connection issues
        if (!axiosError.response) {
          if (!isNetworkIssueReported) {
            console.error('Network Error: Unable to connect to backend API');
            console.error(`Attempted to connect to: ${axiosError.config?.baseURL}${axiosError.config?.url}`);
            isNetworkIssueReported = true;
          }
        } else {
          // Log the error response data
          console.error('API Error Response:', axiosError.response.data);
            
          // Special handling for common error codes
          if (axiosError.response.status === 405) {
            console.error('Method Not Allowed: The API endpoint does not support this operation');
          } else if (axiosError.response.status === 413) {
            console.error('Payload Too Large: The file is too large for the server to accept');
          } else if (axiosError.response.status === 429) {
            console.error('Too Many Requests: Rate limit exceeded');
          }
        }
      } else {
        console.error('Unexpected Error:', error);
      }
      
      // Re-throw the error for the caller to handle
      return Promise.reject(error);
    }
  );
};

// Initialize the API client if in browser
if (isBrowser()) {
  initializeApi();
}

/**
 * Interface for query request parameters
 * Updated to support Ragie-specific features
 */
export interface QueryRequest {
  query: string;
  document_ids?: string[];
  metadata_filter?: Record<string, any>;
  rerank?: boolean;
  top_k?: number;
  show_timings?: boolean;
}

/**
 * Interface for query response
 * Updated to support Ragie's more detailed response format
 */
export interface QueryResponse {
  query: string;
  response: string;
  chunks?: Array<{
    text: string;
    score: number;
    metadata: Record<string, any>;
    document_id: string;
  }>;
  document_ids?: string[];
  timings?: Record<string, number>;
}

/**
 * Interface for document upload result
 */
export interface DocumentUploadResult {
  message: string;
  document_id?: string;
  ragie_document_id?: string;
  status?: string;
  metadata?: Record<string, any>;
}

/**
 * Interface for batch upload result item
 */
export interface BatchUploadResultItem {
  id: string;
  filename: string;
  status: 'success' | 'error' | 'processing';
  message?: string;
  ragie_document_id?: string;
  details?: any;
}

/**
 * Function to retry API calls with exponential backoff
 */
async function withRetry<T>(fn: () => Promise<T>, retries = 2, delay = 1000): Promise<T> {
  try {
    return await fn();
  } catch (error) {
    if (retries === 0) {
      throw error;
    }
    
    console.log(`API call failed, retrying in ${delay}ms (${retries} retries left)`);
    await new Promise(resolve => setTimeout(resolve, delay));
    
    return withRetry(fn, retries - 1, delay * 2);
  }
}

/**
 * Function to check document status
 */
async function checkDocumentStatus(documentId: string): Promise<{ status: string; message: string; document_id: string }> {
  if (!isBrowser() || !api) {
    throw new Error('API client not available in this environment');
  }

  try {
    // Use the correct endpoint with the document ID
    const response = await api.get(`/api/documents/${documentId}/status`);
    
    if (response.status >= 200 && response.status < 300) {
    return response.data;
    } else {
      throw new Error(`Unexpected response status: ${response.status}`);
    }
  } catch (error) {
    console.error(`Error checking document status for ${documentId}:`, error);
    throw new Error(`Failed to check document status`);
  }
}

/**
 * Function to upload a document
 */
async function uploadDocument(
  file: File,
  metadata?: { title?: string; author?: string; description?: string },
  onProgress?: (percentage: number) => void
): Promise<DocumentUploadResult> {
  if (!isBrowser() || !api) {
    throw new Error('API client not available in this environment');
  }

  // Log the upload attempt
  console.log('Attempting document upload:', {
    filename: file.name,
    fileSize: `${(file.size / 1024 / 1024).toFixed(2)} MB`,
    fileType: file.type,
    hasMetadata: !!metadata
  });

  const formData = new FormData();
  formData.append('file', file);
  
  // Add metadata fields if provided
  if (metadata?.title) formData.append('title', metadata.title);
  if (metadata?.author) formData.append('author', metadata.author);
  if (metadata?.description) formData.append('description', metadata.description);

  try {
    // Use the correct endpoint - the backend doesn't use /api prefix
    const response = await api.post('/documents', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const percentage = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          console.log('Upload progress:', percentage + '%', { 
            loaded: progressEvent.loaded,
            total: progressEvent.total
          });
          onProgress?.(percentage);
        }
      },
    });
    
    if (response.status >= 200 && response.status < 300) {
      return response.data;
    } else {
      throw new Error(`Unexpected response status: ${response.status}`);
    }
  } catch (error: any) {
    // More detailed error handling based on error type
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;
      
      // Handle specific status codes with user-friendly messages
      if (axiosError.response?.status === 405) {
        console.error('Method Not Allowed: The server does not support file uploads at this endpoint');
        throw new Error('Upload API not available. Server does not accept document uploads.');
      } else if (!axiosError.response && axiosError.message.includes('timeout')) {
        console.error('Upload timeout: The request took too long to complete');
        throw new Error('Upload timed out. The file may be too large or the server is busy.');
      } else if (axiosError.response?.status === 413) {
        throw new Error('The file is too large for the server to accept. Maximum size is 50MB.');
      }
    }
    
    // Re-throw the error with more context
    throw new Error(`Error uploading document: ${error.message || 'Unknown error'}`);
  }
}

/**
 * Function to upload multiple documents
 */
async function uploadMultipleDocuments(
  files: File[],
  metadata?: { titlePrefix?: string; author?: string; description?: string },
  onProgress?: (percentage: number) => void
): Promise<{ 
  message: string; 
  successful_count: number; 
  failed_count: number; 
  results: BatchUploadResultItem[] 
}> {
  if (!isBrowser() || !api) {
    throw new Error('API client not available in this environment');
  }

  // Log the batch upload attempt
  console.log('Attempting batch document upload:', {
    fileCount: files.length,
    totalSize: `${(files.reduce((acc, f) => acc + f.size, 0) / 1024 / 1024).toFixed(2)} MB`,
    fileTypes: [...new Set(files.map(f => f.type))],
    hasMetadata: !!metadata
  });
  
  if (files.length === 0) {
    return {
      message: 'No files to upload',
      successful_count: 0,
      failed_count: 0,
      results: []
    };
  }

  const formData = new FormData();
  
  // Add each file to the form data
  files.forEach(file => {
    formData.append('files', file);
  });
  
  // Add metadata fields if provided
  if (metadata?.titlePrefix) formData.append('title_prefix', metadata.titlePrefix);
  if (metadata?.author) formData.append('author', metadata.author);
  if (metadata?.description) formData.append('description', metadata.description);
  
  try {
    // Use the correct endpoint - the backend doesn't use /api prefix
    const response = await api.post('/documents/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const percentage = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          console.log('Batch upload progress:', percentage + '%', { 
            loaded: progressEvent.loaded,
            total: progressEvent.total
          });
          onProgress?.(percentage);
        }
      },
    });
    
    if (response.status >= 200 && response.status < 300) {
      return response.data;
    } else {
      throw new Error(`Unexpected response status: ${response.status}`);
    }
  } catch (error: any) {
    // More detailed error handling
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;
      
      // Handle specific status codes with user-friendly messages
      if (axiosError.response?.status === 405) {
        console.error('Method Not Allowed: The server does not support batch file uploads');
        throw new Error('Batch upload API not available. Server does not accept multiple document uploads.');
      } else if (!axiosError.response && axiosError.message.includes('timeout')) {
        console.error('Batch upload timeout: The request took too long to complete');
        throw new Error('Batch upload timed out. The files may be too large or the server is busy.');
      } else if (axiosError.response?.status === 413) {
        throw new Error('The batch is too large for the server to accept. Try fewer or smaller files.');
      }
    }
    
    // Re-throw the error with more context
    throw new Error(`Error uploading documents: ${error.message || 'Unknown error'}`);
  }
}

/**
 * Function to check health of the backend API
 */
async function checkHealth(): Promise<{ message: string }> {
  if (!isBrowser() || !api) {
    throw new Error('API client not available in this environment');
  }

  const healthEndpoint = process.env.NODE_ENV === 'production' ? '/health' : '/health';
  
  try {
    console.log(`Checking API health at endpoint: ${healthEndpoint}`);
    const response = await api.get(healthEndpoint, { 
      timeout: 5000 // Short timeout for health checks
    });
    
    console.log('Health check response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw new Error('API health check failed. The backend may be unavailable.');
  }
}

/**
 * Function to list all documents
 */
async function listDocuments(): Promise<Array<{ id: string; name: string; status: string; metadata: Record<string, any> }>> {
  if (!isBrowser() || !api) {
    throw new Error('API client not available in this environment');
  }

  try {
    // Use the correct endpoint - the backend uses /api/documents
    const response = await api.get('/api/documents');
    
    if (response.status >= 200 && response.status < 300) {
    return response.data;
    } else {
      throw new Error(`Unexpected response status: ${response.status}`);
    }
  } catch (error) {
    console.error('Error listing documents:', error);
    throw new Error('Failed to retrieve document list');
  }
}

/**
 * Function to delete a document
 */
async function deleteDocument(documentId: string): Promise<{ status: string; message: string }> {
  if (!isBrowser() || !api) {
    throw new Error('API client not available in this environment');
  }

  try {
    // Use the correct endpoint with the document ID
    const response = await api.delete(`/api/documents/${documentId}`);
    
    if (response.status >= 200 && response.status < 300) {
    return response.data;
    } else {
      throw new Error(`Unexpected response status: ${response.status}`);
    }
  } catch (error) {
    console.error(`Error deleting document ${documentId}:`, error);
    throw new Error('Failed to delete document');
  }
}

export const ragApi = {
  /**
   * Send a query to the RAG system
   */
  async query(queryRequest: QueryRequest): Promise<QueryResponse> {
    if (!isBrowser() || !api) {
      throw new Error('API client not available in this environment');
    }

    // Log query details
    console.log('Sending query request:', { 
      query: queryRequest.query.length > 50 ? queryRequest.query.substring(0, 50) + '...' : queryRequest.query,
      docCount: queryRequest.document_ids ? queryRequest.document_ids.length : 'all',
      rerank: queryRequest.rerank,
      topK: queryRequest.top_k
    });

    try {
      // Use the correct endpoint with /api prefix as defined in the backend
      const response = await api.post('/api/query', queryRequest);
      
      if (response.status >= 200 && response.status < 300) {
        return response.data;
      } else {
        throw new Error(`Unexpected response status: ${response.status}`);
      }
    } catch (error: any) {
      // Enhanced error handling
      if (axios.isAxiosError(error)) {
        const axiosError = error as AxiosError;
        
        if (axiosError.response?.status === 405) {
          console.error('Method Not Allowed: The server does not support this query endpoint');
          throw new Error('Query API not available. Server does not accept queries at this endpoint.');
        } else if (!axiosError.response && axiosError.message.includes('timeout')) {
          console.error('Query timeout: The request took too long to complete');
          throw new Error('Query timed out. The server may be busy processing a complex query.');
        }
      }
      
      // Re-throw the error with more context
      throw new Error(`Error querying documents: ${error.message || 'Unknown error'}`);
    }
  },
  
  /**
   * Upload a document to the RAG system
   */
  async uploadDocument(
    file: File,
    metadata?: { title?: string; author?: string; description?: string },
    onProgress?: (percentage: number) => void
  ): Promise<DocumentUploadResult> {
    return uploadDocument(file, metadata, onProgress);
  },
  
  /**
   * Upload multiple documents to the RAG system
   */
  async uploadMultipleDocuments(
    files: File[],
    metadata?: { titlePrefix?: string; author?: string; description?: string },
    onProgress?: (percentage: number) => void
  ): Promise<{ 
    message: string; 
    successful_count: number; 
    failed_count: number; 
    results: BatchUploadResultItem[] 
  }> {
    return uploadMultipleDocuments(files, metadata, onProgress);
  },
  
  /**
   * Check the health of the RAG system
   */
  async checkHealth(): Promise<{ message: string }> {
    return checkHealth();
  },
  
  /**
   * Check the status of a document
   */
  async checkDocumentStatus(documentId: string): Promise<{ status: string; message: string; document_id: string }> {
    return checkDocumentStatus(documentId);
  },
  
  /**
   * List all documents
   */
  async listDocuments(): Promise<Array<{ id: string; name: string; status: string; metadata: Record<string, any> }>> {
    return listDocuments();
  },
  
  /**
   * Delete a document
   */
  async deleteDocument(documentId: string): Promise<{ status: string; message: string }> {
    return deleteDocument(documentId);
  }
}; 