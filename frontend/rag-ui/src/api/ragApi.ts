import axios, { AxiosError } from 'axios';
import config from '../config';

// Create an axios instance with default config
const api = axios.create({
  baseURL: config.apiUrl,
  headers: {
    'Content-Type': 'application/json',
  },
  // Add timeout to prevent hanging requests
  timeout: 30000, // 30 seconds
});

// Add request interceptor for logging
api.interceptors.request.use(request => {
  console.log(`API Request: ${request.method?.toUpperCase()} ${request.baseURL}${request.url}`);
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
      
      console.error('API Error:', {
        message: axiosError.message,
        code: axiosError.code,
        status: axiosError.response?.status,
        statusText: axiosError.response?.statusText,
        url: axiosError.config?.url
      });
      
      // Customize error message based on error type
      if (!axiosError.response) {
        error.message = 'Network error: Please check your connection and ensure the backend server is running.';
      } else if (axiosError.response.status === 404) {
        error.message = 'API endpoint not found. Please check the URL and server configuration.';
      } else if (axiosError.response.status === 403) {
        error.message = 'Access forbidden. You might need to authenticate or check permissions.';
      } else if (axiosError.response.status >= 500) {
        error.message = 'Server error. Please try again later or contact support.';
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
      const response = await api.post<QueryResponse>('/query', queryRequest);
      return response.data;
    } catch (error) {
      console.error('Query error:', error);
      throw error;
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
  
  async uploadMultipleDocuments(
    files: File[],
    metadata?: { titlePrefix?: string; author?: string; description?: string }
  ): Promise<{ message: string; successful_count: number; failed_count: number; results: any[] }> {
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