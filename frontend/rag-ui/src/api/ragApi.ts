import axios, { AxiosError } from 'axios';
import config from '../config';

// Network health check state
let isNetworkIssueReported = false;

// Create an axios instance with default config
const api = axios.create({
  baseURL: config.apiUrl,
  headers: {
    'Content-Type': 'application/json',
  },
  // Add timeout to prevent hanging requests
  timeout: 30000, // 30 seconds
  // Add better retry behavior
  validateStatus: status => status < 500, // Treat 500+ as errors
});

// Add request interceptor for logging
api.interceptors.request.use(request => {
  console.log(`API Request: ${request.method?.toUpperCase()} ${request.baseURL}${request.url}`);
  
  // Reset network issue flag for new requests
  if (request.url !== '/health') {
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
        url: axiosError.config?.url
      });
      
      // Check for backend connection issues
      if (!axiosError.response) {
        // Only report network issue once to avoid console spam
        if (!isNetworkIssueReported) {
          console.error('Backend connection issue detected. This may indicate that the server is down, overloaded, or experiencing network issues.');
          isNetworkIssueReported = true;
        }
        
        // Provide detailed troubleshooting steps in the error message
        error.message = `Network error: Please check your connection and ensure the backend server is running.\n\nTroubleshooting steps:\n1. Verify the backend server is running (command: python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000)\n2. Check if API URL (${config.apiUrl}) is correct in .env.local\n3. Look for CORS errors in browser dev tools`;
      } else if (axiosError.response.status === 404) {
        error.message = `API endpoint not found: ${axiosError.config?.url}. Please check the URL and server configuration.`;
      } else if (axiosError.response.status === 403) {
        error.message = 'Access forbidden. You might need to authenticate or check permissions.';
      } else if (axiosError.response.status >= 500) {
        error.message = `Server error (${axiosError.response.status}): ${axiosError.response.statusText || 'Unknown server error'}. Please try again later or check server logs.`;
      }
    }
    
    return Promise.reject(error);
  }
);

/**
 * Interface for query request
 */
export interface QueryRequest {
  query: string;
  filters?: Record<string, unknown>;
}

/**
 * Interface for query response
 */
export interface QueryResponse {
  answer: string;
}

/**
 * Interface for document upload result
 */
interface DocumentUploadResult {
  id: string;
  filename: string;
  status: 'success' | 'error';
  message?: string;
}

/**
 * Utility function to retry API calls
 * @param fn Function to retry
 * @param retries Number of retries
 * @param delay Delay between retries in ms
 */
async function withRetry<T>(fn: () => Promise<T>, retries = 2, delay = 1000): Promise<T> {
  try {
    return await fn();
  } catch (error) {
    if (retries <= 0) throw error;
    
    console.log(`Retrying in ${delay}ms... (${retries} attempts left)`);
    await new Promise(resolve => setTimeout(resolve, delay));
    return withRetry(fn, retries - 1, delay * 1.5);
  }
}

/**
 * RAG API client
 */
export const ragApi = {
  /**
   * Query the RAG system
   * @param queryRequest - Query request
   * @returns Query response
   */
  async query(queryRequest: QueryRequest): Promise<QueryResponse> {
    try {
      // Use a longer timeout for queries since they require LLM processing
      const response = await withRetry(async () => {
        console.log('Sending query to backend:', queryRequest.query.substring(0, 50) + (queryRequest.query.length > 50 ? '...' : ''));
        return api.post<QueryResponse>('/query', queryRequest, {
          timeout: 120000, // 2 minutes timeout for LLM processing
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          }
        });
      }, 2, 2000); // Retry twice with 2s initial delay
      
      console.log('Query response received successfully');
      return response.data;
    } catch (error) {
      let errorMessage = 'Failed to get a response from the backend.';
      
      if (axios.isAxiosError(error)) {
        const axiosError = error as AxiosError;
        
        if (!axiosError.response) {
          // Network error or timeout
          errorMessage = 'The query request timed out or failed to reach the server. This is often due to the LLM processing taking too long. Try a simpler query or check if the server is running.';
        } else if (axiosError.response.status === 400) {
          // Bad request
          errorMessage = 'Invalid query format. Please check your input and try again.';
        } else if (axiosError.response.status === 422) {
          // Validation error
          errorMessage = 'The server could not process your query. It may be too complex or contain invalid characters.';
        } else if (axiosError.response.status >= 500) {
          // Server error
          errorMessage = 'The server encountered an error while processing your query. This might be due to heavy load or issues with the LLM.';
        }
        
        // Add detailed error info for debugging
        console.error('Detailed query error:', {
          status: axiosError.response?.status,
          statusText: axiosError.response?.statusText,
          data: axiosError.response?.data,
          message: axiosError.message,
          url: axiosError.config?.url,
          method: axiosError.config?.method
        });
      }
      
      console.error('Query error:', errorMessage);
      
      // Create a custom error with our message
      const customError = new Error(errorMessage);
      Object.assign(customError, error); // Keep original error properties
      throw customError;
    }
  },

  /**
   * Upload a document to the RAG system
   * @param file - Document file
   * @param metadata - Document metadata
   * @returns Upload result
   */
  async uploadDocument(
    file: File,
    metadata?: { title?: string; author?: string; description?: string }
  ): Promise<{ message: string }> {
    const formData = new FormData();
    formData.append('file', file);
    
    if (metadata?.title) {
      formData.append('title', metadata.title);
    }
    
    if (metadata?.author) {
      formData.append('author', metadata.author);
    }
    
    if (metadata?.description) {
      formData.append('description', metadata.description);
    }
    
    try {
      const response = await api.post('/documents', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      return response.data;
    } catch (error) {
      console.error('Error uploading document:', error);
      throw error;
    }
  },
  
  /**
   * Upload multiple documents to the RAG system
   * @param files - Array of document files
   * @param metadata - Document metadata
   * @returns Upload result with batch statistics
   */
  async uploadMultipleDocuments(
    files: File[],
    metadata?: { titlePrefix?: string; author?: string; description?: string }
  ): Promise<{ message: string; successful_count: number; failed_count: number; results: DocumentUploadResult[] }> {
    const formData = new FormData();
    
    // Append all files
    for (const file of files) {
      formData.append('files', file);
    }
    
    // Add metadata
    if (metadata?.titlePrefix) {
      formData.append('title_prefix', metadata.titlePrefix);
    }
    
    if (metadata?.author) {
      formData.append('author', metadata.author);
    }
    
    if (metadata?.description) {
      formData.append('description', metadata.description);
    }
    
    try {
      const response = await api.post('/documents/batch', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        // Increase timeout for large batch uploads
        timeout: 300000, // 5 minutes
      });
      
      return response.data;
    } catch (error) {
      console.error('Error uploading multiple documents:', error);
      throw error;
    }
  },

  /**
   * Check if the API is available
   * @returns Welcome message
   */
  async checkHealth(): Promise<{ message: string }> {
    try {
      // Use withRetry to automatically retry on network errors
      return await withRetry(async () => {
        console.log('Checking API health...');
        const response = await api.get<{ message: string }>('/health');
        console.log('API health check successful');
        return response.data;
      }, 2, 1000);
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  },
}; 