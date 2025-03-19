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

  // Create an axios instance with default config
  api = axios.create({
    baseURL: config.apiUrl,
    headers: {
      'Content-Type': 'application/json',
    },
    // Add timeout to prevent hanging requests
    timeout: 300000, // 5 minutes
    // Add better retry behavior
    validateStatus: status => status < 500, // Treat 500+ as errors
  });

  // Log the API configuration
  console.log('API Client Configuration:', {
    baseURL: api.defaults.baseURL,
    environment: process.env.NODE_ENV,
    version: config.version
  });

  // Safety check - we should never use localhost in production
  if (process.env.NODE_ENV === 'production' && api.defaults.baseURL?.includes('localhost')) {
    console.error('CRITICAL ERROR: Using localhost in production environment!');
    // Force override to production URL
    api.defaults.baseURL = 'https://rag-bpql.onrender.com';
    console.log('Forced API URL to:', api.defaults.baseURL);
  }

  // Fix URL paths for API calls
  const resolveApiPath = (path: string) => {
    // Ensure path starts with a slash if not empty
    if (path && !path.startsWith('/')) {
      path = '/' + path;
    }
    return path;
  };

  // Add request interceptor for logging
  api.interceptors.request.use(request => {
    // Fix URL paths to ensure they're properly formatted
    if (request.url) {
      request.url = resolveApiPath(request.url);
    }

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
          if (!isNetworkIssueReported) {
            console.error('Network Error: Unable to connect to backend API');
            isNetworkIssueReported = true;
          }
        } else {
          // Log the error response data
          console.error('API Error Response:', axiosError.response.data);
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
    throw new Error('API client is not available in server environment');
  }
  
  // Store a reference to the API instance to satisfy TypeScript
  const apiClient = api;
  
  try {
    const response = await apiClient.get(`/documents/${documentId}/status`);
    return response.data;
  } catch (error) {
    console.error('Error checking document status:', error);
    throw new Error('Unable to check document status');
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
    throw new Error('API client is not available in server environment');
  }

  // Store a reference to the API instance to satisfy TypeScript
  const apiClient = api;

  const formData = new FormData();
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
    // Calculate timeout based on file size for large files
    const timeout = Math.min(
      config.baseTimeout + (file.size / 1024 / 1024) * config.timeoutPerMb,
      config.maxTimeout
    );
    
    // Report initial upload starting
    if (onProgress) {
      onProgress(0);
    }
    
    // Set up the upload request with progress tracking
    const response = await apiClient.post<DocumentUploadResult>('/documents', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout,
      onUploadProgress: progressEvent => {
        if (progressEvent.total && onProgress) {
          // Calculate upload progress (0-80%)
          const percentage = Math.min(
            80, // Cap at 80% to show that processing still needs to happen
            Math.round((progressEvent.loaded / progressEvent.total) * 100 * 0.8)
          );
          onProgress(percentage);
        }
      },
    });
    
    // Function to handle document status polling
    const pollStatus = async () => {
      try {
        let processingComplete = false;
        let retries = 10;
        let delayMs = 1000;
        let currentProgress = 80;
        
        const documentId = response.data.document_id;
        if (!documentId) {
          throw new Error("No document ID returned from server");
        }
        
        while (!processingComplete && retries > 0) {
          // Wait before checking status
          await new Promise(resolve => setTimeout(resolve, delayMs));
          
          // Check document status
          const statusResponse = await checkDocumentStatus(documentId);
          console.log(`Document status: ${JSON.stringify(statusResponse)}`);
          
          // Update progress based on status
          if (statusResponse.status === 'complete') {
            if (onProgress) {
              onProgress(100);
            }
            processingComplete = true;
          } else if (statusResponse.status === 'error') {
            throw new Error(`Error processing document: ${statusResponse.message}`);
          } else if (statusResponse.status === 'processing') {
            // Increment progress for processing (80-95%)
            if (onProgress && currentProgress < 95) {
              currentProgress += 1.5;
              onProgress(Math.round(currentProgress));
            }
            
            // Increase delay for next check
            delayMs = Math.min(delayMs * 1.5, 10000);
            retries--;
          } else if (statusResponse.status === 'timeout') {
            // Document is still processing but taking longer than expected
            if (onProgress) {
              onProgress(95); // Cap at 95%
            }
            
            // Increase delay significantly
            delayMs = 10000;
            retries--;
          }
        }
        
        if (!processingComplete) {
          // We've polled several times but the document is still processing
          // Return success but note that processing continues
          return {
            message: 'Document uploaded and is still being processed. It will be available for querying soon.',
            document_id: documentId,
            ragie_document_id: response.data.metadata?.ragie_document_id
          };
        }
        
        return {
          message: 'Document uploaded and processed successfully.',
          document_id: documentId,
          ragie_document_id: response.data.metadata?.ragie_document_id
        };
      } catch (error) {
        console.error('Error during document status polling:', error);
        throw error;
      }
    };
    
    if (response.status === 202) {
      // Document accepted for processing, start polling for status
      return await pollStatus();
    } else {
      throw new Error(`Unexpected response status: ${response.status}`);
    }
  } catch (error) {
    console.error('Error uploading document:', error);
    if (onProgress) {
      onProgress(0);
    }
    throw error;
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
    throw new Error('API client is not available in server environment');
  }

  // Store a reference to the API instance to satisfy TypeScript
  const apiClient = api;

  const formData = new FormData();
  
  // Add all files
  files.forEach(file => {
    formData.append('files', file);
  });
  
  // Add metadata if provided
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
    // Calculate timeout based on combined file size
    const totalSize = files.reduce((sum, file) => sum + file.size, 0);
    const timeout = Math.min(
      config.baseTimeout + (totalSize / 1024 / 1024) * config.timeoutPerMb,
      config.maxTimeout
    );
    
    // Report initial upload starting
    if (onProgress) {
      onProgress(0);
    }
    
    // Set up the upload request with progress tracking
    const response = await apiClient.post('/documents/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout,
      onUploadProgress: progressEvent => {
        if (progressEvent.total && onProgress) {
          // Calculate upload progress (0-70%)
          const percentage = Math.min(
            70, // Cap at 70% to show that processing still needs to happen
            Math.round((progressEvent.loaded / progressEvent.total) * 100 * 0.7)
          );
          onProgress(percentage);
        }
      },
    });
    
    if (response.status === 202) {
      // Documents accepted for processing
      const results = response.data.results || [];
      
      // Start polling for status if we have results
      if (results.length > 0) {
        const pollStatus = async () => {
          try {
            let allProcessingComplete = false;
            let retries = 10;
            let delayMs = 2000;
            let currentProgress = 70;
            
            // Get document IDs to poll
            const documentIds = results
              .filter((result: BatchUploadResultItem) => result.status === 'processing')
              .map((result: BatchUploadResultItem) => result.id);
            
            while (!allProcessingComplete && retries > 0 && documentIds.length > 0) {
              // Wait before checking status
              await new Promise(resolve => setTimeout(resolve, delayMs));
              
              // Check each document status
              let pendingCount = 0;
              
              for (const docId of documentIds) {
                try {
                  const statusResponse = await checkDocumentStatus(docId);
                  
                  // Update the status in results
                  const resultIndex = results.findIndex((r: BatchUploadResultItem) => r.id === docId);
                  if (resultIndex >= 0) {
                    // Update status
                    if (statusResponse.status === 'complete') {
                      results[resultIndex].status = 'success';
                      results[resultIndex].message = 'Document processed successfully';
                    } else if (statusResponse.status === 'error') {
                      results[resultIndex].status = 'error';
                      results[resultIndex].message = statusResponse.message;
                    } else {
                      results[resultIndex].status = 'processing';
                      pendingCount++;
                    }
                  }
                } catch (error) {
                  console.error(`Error checking status for document ${docId}:`, error);
                  pendingCount++;
                }
              }
              
              // Update progress for processing (70-95%)
              if (onProgress && pendingCount === 0) {
                onProgress(100);
                allProcessingComplete = true;
              } else if (onProgress && currentProgress < 95) {
                // Calculate progress based on remaining pending docs
                const progressIncrement = (95 - currentProgress) / (retries + 1);
                currentProgress += progressIncrement;
                onProgress(Math.round(currentProgress));
              }
              
              if (pendingCount === 0) {
                allProcessingComplete = true;
              } else {
                // Increase delay for next check
                delayMs = Math.min(delayMs * 1.5, 10000);
                retries--;
              }
            }
            
            // Count success and failures
            const successCount = results.filter((r: BatchUploadResultItem) => r.status === 'success').length;
            const failCount = results.filter((r: BatchUploadResultItem) => r.status === 'error').length;
            const stillProcessingCount = results.filter((r: BatchUploadResultItem) => r.status === 'processing').length;
            
            if (stillProcessingCount > 0) {
              return {
                message: `${successCount} documents processed successfully, ${failCount} failed, ${stillProcessingCount} still processing in the background.`,
                successful_count: successCount,
                failed_count: failCount,
                results
              };
            }
            
            return {
              message: `${successCount} documents processed successfully, ${failCount} failed.`,
              successful_count: successCount,
              failed_count: failCount,
              results
            };
          } catch (error) {
            console.error('Error during batch status polling:', error);
            throw error;
          }
        };
        
        return await pollStatus();
      }
      
      return response.data;
    } else {
      throw new Error(`Unexpected response status: ${response.status}`);
    }
  } catch (error) {
    console.error('Error uploading multiple documents:', error);
    if (onProgress) {
      onProgress(0);
    }
    throw error;
  }
}

/**
 * Function to check health of the backend API
 */
async function checkHealth(): Promise<{ message: string }> {
  if (!isBrowser() || !api) {
    return { message: 'API health check not available in server environment' };
  }
  
  // Store a reference to the API instance to satisfy TypeScript
  const apiClient = api;
  
  try {
    // Set a shorter timeout for health check
    const response = await apiClient.get('/health', { timeout: 5000 });
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    return { message: 'API health check failed. Backend may be unavailable.' };
  }
}

/**
 * Function to list all documents
 */
async function listDocuments(): Promise<Array<{ id: string; name: string; status: string; metadata: Record<string, any> }>> {
  if (!isBrowser() || !api) {
    throw new Error('API client is not available in server environment');
  }
  
  // Store a reference to the API instance to satisfy TypeScript
  const apiClient = api;
  
  try {
    const response = await apiClient.get('/documents');
    return response.data;
  } catch (error) {
    console.error('Error listing documents:', error);
    throw new Error('Unable to fetch documents');
  }
}

/**
 * Function to delete a document
 */
async function deleteDocument(documentId: string): Promise<{ status: string; message: string }> {
  if (!isBrowser() || !api) {
    throw new Error('API client is not available in server environment');
  }
  
  // Store a reference to the API instance to satisfy TypeScript
  const apiClient = api;
  
  try {
    const response = await apiClient.delete(`/documents/${documentId}`);
    return response.data;
  } catch (error) {
    console.error(`Error deleting document ${documentId}:`, error);
    throw new Error('Unable to delete document');
  }
}

export const ragApi = {
  /**
   * Send a query to the RAG system
   */
  async query(queryRequest: QueryRequest): Promise<QueryResponse> {
    if (!isBrowser() || !api) {
      throw new Error('API client is not available in server environment');
    }
    
    // Store a reference to the API instance to satisfy TypeScript
    const apiClient = api;
    
    try {
      // Use the new API endpoint
      const response = await withRetry(() => apiClient.post('/query', queryRequest));
      
      if (response.status === 200) {
        return response.data;
      } else {
        throw new Error(`Unexpected response status: ${response.status}`);
      }
    } catch (error) {
      console.error('Error querying:', error);
      throw error;
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