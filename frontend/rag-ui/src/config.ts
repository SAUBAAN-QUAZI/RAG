/**
 * Frontend configuration settings
 */

// Helper to check if we're running in browser
const isBrowser = typeof window !== 'undefined';

// Default API URL (fallback if not set in .env.local)
const DEFAULT_API_URL = 'http://localhost:8000';

// Environment variable parsing helper with logging
const getEnvVar = (key: string, defaultValue: string): string => {
  const envKey = `NEXT_PUBLIC_${key}`;
  const value = process.env[envKey];
  
  // Only log in browser to avoid SSR logs
  if (isBrowser) {
    console.log(`Config: Reading ${envKey}=${value || '(not set, using default)'}`);
  }
  
  return value || defaultValue;
};

// Parse boolean environment variable
const getEnvBool = (key: string, defaultValue: boolean): boolean => {
  const value = process.env[`NEXT_PUBLIC_${key}`];
  if (value === 'true') return true;
  if (value === 'false') return false;
  return defaultValue;
};

// Parse numeric environment variable
const getEnvNumber = (key: string, defaultValue: number): number => {
  const value = process.env[`NEXT_PUBLIC_${key}`];
  if (value) {
    const parsedValue = parseInt(value, 10);
    if (!isNaN(parsedValue)) return parsedValue;
  }
  return defaultValue;
};

// Get the API URL with runtime verification
const getApiUrl = (): string => {
  // Try to get from runtime environment
  const apiUrl = getEnvVar('API_URL', DEFAULT_API_URL);
  
  // Force override in production to prevent localhost
  if (isBrowser && process.env.NODE_ENV === 'production' && apiUrl.includes('localhost')) {
    console.warn('CONFIG WARNING: Using localhost in production! Forcing to use Render backend');
    return 'https://rag-bpql.onrender.com';
  }
  
  // Only log in browser to avoid SSR logs
  if (isBrowser) {
    console.log(`Config: Using API URL: ${apiUrl}`);
  }
  
  return apiUrl;
};

/**
 * Application configuration
 */
const config = {
  // API configuration
  apiUrl: getApiUrl(),
  
  // Version (for cache busting)
  version: getEnvVar('VERSION', '1.0.0'),
  
  // Upload configuration
  maxFileSize: getEnvNumber('MAX_FILE_SIZE', 52428800), // 50MB default
  maxBatchSize: getEnvNumber('MAX_BATCH_SIZE', 5),
  
  // Timeout configuration (milliseconds)
  baseTimeout: 30000, // 30 seconds base timeout
  timeoutPerMb: 5000, // Add 5 seconds per MB of file size
  maxTimeout: 300000, // Maximum timeout (5 minutes)
  
  // Chat configuration
  defaultRerank: getEnvBool('DEFAULT_RERANK', true),
  defaultTopK: getEnvNumber('DEFAULT_TOP_K', 3),
  
  // Feature flags
  enableSearch: getEnvBool('ENABLE_SEARCH', true),
  enableDelete: getEnvBool('ENABLE_DELETE', true),
  
  // Environment
  isDevelopment: process.env.NODE_ENV !== 'production',
};

export default config; 