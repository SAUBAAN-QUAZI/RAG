/**
 * Application configuration
 */
const config = {
  // API URL - Uses environment variable or falls back to localhost for development
  apiUrl: (() => {
    const configuredUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    console.log(`Using API URL: ${configuredUrl}`);
    return configuredUrl;
  })(),
  
  // Default configuration for document processing
  defaultChunkSize: 1000,
  defaultChunkOverlap: 200,
  
  // UI configuration
  maxFileSize: 50 * 1024 * 1024, // 50MB in bytes
  acceptedFileTypes: ['.pdf'],
  
  // Timeout configuration (milliseconds)
  baseTimeout: 30000, // 30 seconds base timeout
  timeoutPerMb: 60000, // Additional 60 seconds per MB
  maxTimeout: 300000, // 5 minute maximum timeout
};

export default config; 