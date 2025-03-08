import axios from 'axios';
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
    // Create a FormData object
    const formData = new FormData();
    
    // Add the file to the form data
    formData.append('file', file);

    // Add metadata if provided
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
      // Log some debug information
      console.log('Uploading file:', file.name, file.size, file.type);
      console.log('API URL:', config.apiUrl);
      
      // Calculate a reasonable timeout based on file size
      // Larger files need more time for processing
      const fileSize = file.size;
      const timeoutPerMB = 60000; // 60 seconds per MB
      const baseTimeout = 30000; // 30 seconds base timeout
      const calculatedTimeout = Math.max(
        baseTimeout,
        Math.min(300000, Math.ceil(fileSize / (1024 * 1024)) * timeoutPerMB) // Cap at 5 minutes
      );
      
      console.log(`Setting timeout to ${calculatedTimeout}ms based on file size ${(fileSize / (1024 * 1024)).toFixed(2)}MB`);
      
      // Make the request with multipart/form-data
      const response = await axios.post(`${config.apiUrl}/documents`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        // Use dynamic timeout based on file size
        timeout: calculatedTimeout,
        // Add progress tracking
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 1));
          console.log(`Upload progress: ${percentCompleted}%`);
        },
      });
      
      return response.data;
    } catch (error) {
      console.error('Upload error:', error);
      
      // Add specific error handling for timeouts
      if (axios.isAxiosError(error) && error.code === 'ECONNABORTED') {
        throw new Error('Upload timed out. The document may be too large or server processing took too long. Try with a smaller document or try again later.');
      }
      
      throw error;
    }
  },

  /**
   * Check if the API is available
   * @returns Welcome message
   */
  async checkHealth(): Promise<{ message: string }> {
    try {
      const response = await api.get<{ message: string }>('/');
      return response.data;
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  },
}; 