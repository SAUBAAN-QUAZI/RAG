/**
 * Test script for configuration
 * 
 * This script checks the configuration setup for development and production environments
 * to verify that API URLs are correctly configured.
 */

// Add a helper to detect environment
const detectEnvironment = () => {
  // Check various environment variables
  if (process.env.NODE_ENV === 'production') return 'production';
  if (process.env.VERCEL_ENV === 'production') return 'production';
  if (process.env.NEXT_PUBLIC_ENVIRONMENT === 'production') return 'production';
  return 'development';
};

// Simulate the browser environment variables for testing
process.env.NEXT_PUBLIC_API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
process.env.NEXT_PUBLIC_VERSION = process.env.NEXT_PUBLIC_VERSION || '1.0.3';

// Output basic environment info
console.log('=== Current Environment ===');
console.log('NEXT_PUBLIC_API_URL:', process.env.NEXT_PUBLIC_API_URL);
console.log('NEXT_PUBLIC_VERSION:', process.env.NEXT_PUBLIC_VERSION);
console.log('NODE_ENV:', process.env.NODE_ENV);
console.log('Detected environment:', detectEnvironment());
console.log();

// Simulate development environment
console.log('=== Simulating Development Environment ===');
process.env.NODE_ENV = 'development';
process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8000';
// Load config to test development settings
delete require.cache[require.resolve('./src/config')];
const devConfig = require('./src/config').default;
console.log('API URL:', devConfig.apiUrl);
console.log('Environment:', devConfig.environment);
console.log('Is Production:', devConfig.isProduction);
console.log();

// Simulate production environment
console.log('=== Simulating Production Environment ===');
process.env.NODE_ENV = 'production';
process.env.NEXT_PUBLIC_API_URL = 'https://rag-bpql.onrender.com';
// Reload config to test production settings
delete require.cache[require.resolve('./src/config')];
const prodConfig = require('./src/config').default;
console.log('API URL:', prodConfig.apiUrl);
console.log('Environment:', prodConfig.environment);
console.log('Is Production:', prodConfig.isProduction);
console.log();

// Test localhost in production (should force to production URL)
console.log('=== Testing Production Safety Check ===');
process.env.NODE_ENV = 'production';
process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8000';
// Reload config to test safety mechanism
delete require.cache[require.resolve('./src/config')];
const safetyConfig = require('./src/config').default;
console.log('API URL with localhost in production:', safetyConfig.apiUrl);
console.log('(Should be forced to production URL)');

// Print summary
console.log('\n=== Configuration Test Summary ===');
console.log('Development API URL:', devConfig.apiUrl);
console.log('Production API URL:', prodConfig.apiUrl);
console.log('Safety check working:', safetyConfig.apiUrl !== 'http://localhost:8000' ? 'YES ✅' : 'NO ❌'); 