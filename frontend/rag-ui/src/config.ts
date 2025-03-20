/**
 * Frontend configuration settings
 * 
 * This module provides configuration values for the application,
 * with appropriate fallbacks and environment-specific settings.
 */

// Helper to check if we're running in browser
const isBrowser = typeof window !== 'undefined';

// Production backend URL (Render)
const PRODUCTION_API_URL = 'https://rag-bpql.onrender.com';

// Default API URL (fallback if not set in .env files)
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

// Determine if we're in production environment
const isProduction = (): boolean => {
  if (process.env.NODE_ENV === 'production') return true;
  if (process.env.VERCEL_ENV === 'production') return true;
  if (process.env.NEXT_PUBLIC_ENVIRONMENT === 'production') return true;
  return false;
};

// Get the API URL with runtime verification
const getApiUrl = (): string => {
  // Try to get from runtime environment
  const configuredApiUrl = getEnvVar('API_URL', DEFAULT_API_URL);
  
  // Determine if we're in production
  const isProd = isProduction();
  
  // In production, ensure we're using the production API
  if (isProd) {
    // If API URL is set to localhost in production, override with production URL
    if (configuredApiUrl.includes('localhost')) {
      console.warn('CONFIG WARNING: Using localhost in production environment!');
      console.warn(`Overriding localhost URL with production URL: ${PRODUCTION_API_URL}`);
      return PRODUCTION_API_URL;
    }
    
    // Validate that the URL looks like a proper production URL
    if (!configuredApiUrl.includes('render.com') && 
        !configuredApiUrl.includes('vercel.app') &&
        !configuredApiUrl.startsWith('https://')) {
      console.warn('CONFIG WARNING: API URL may not be configured correctly:', configuredApiUrl);
      console.warn(`Expected a valid HTTPS URL in production. Using fallback: ${PRODUCTION_API_URL}`);
      return PRODUCTION_API_URL;
    }
  }
  
  // Only log in browser to avoid SSR logs
  if (isBrowser) {
    // Add detailed API URL logging to help with debugging
    console.log(`Config: Using API URL: ${configuredApiUrl}`, {
      environment: process.env.NODE_ENV,
      isProduction: isProd,
      origin: typeof window !== 'undefined' ? window.location.origin : 'unknown',
      isLocalhost: configuredApiUrl.includes('localhost'),
    });
  }
  
  return configuredApiUrl;
};

/**
 * Application configuration
 */
const config = {
  // API configuration
  apiUrl: getApiUrl(),
  
  // Environment information
  environment: process.env.NODE_ENV || 'development',
  isProduction: isProduction(),
  
  // Version (for cache busting)
  version: getEnvVar('VERSION', '1.0.3'),
  
  // Upload configuration
  maxFileSize: getEnvNumber('MAX_FILE_SIZE', 52428800), // 50MB default
  maxBatchSize: getEnvNumber('MAX_BATCH_SIZE', 5),
  
  // Timeout configuration (milliseconds)
  baseTimeout: 30000, // 30 seconds base timeout
  timeoutPerMb: 180000, // Add 5 seconds per MB of file size
  maxTimeout: 300000, // Maximum timeout (5 minutes)
  
  // Chat configuration
  defaultRerank: getEnvBool('DEFAULT_RERANK', true),
  defaultTopK: getEnvNumber('DEFAULT_TOP_K', 3),
  
  // Feature flags
  enableSearch: getEnvBool('ENABLE_SEARCH', true),
  enableDelete: getEnvBool('ENABLE_DELETE', true),
  
  // Helper for development mode checking
  isDevelopment: process.env.NODE_ENV !== 'production',
};

export default config; 