"use client";

import { useEffect, useState } from 'react';
import config from '../config';

/**
 * Component to verify configuration at runtime
 * Displays warnings if using incorrect API URL in production
 */
export default function ConfigCheck() {
  const [hasConfigError, setHasConfigError] = useState(false);
  const [configDetails, setConfigDetails] = useState<{
    apiUrl: string;
    environment: string;
    isCorrect: boolean;
  }>({ apiUrl: '', environment: '', isCorrect: true });
  
  useEffect(() => {
    // Only run on client side
    if (typeof window === 'undefined') return;
    
    // Check if we're in production and using localhost
    const isProd = process.env.NODE_ENV === 'production';
    const apiUrl = config.apiUrl;
    const isLocalhost = apiUrl.includes('localhost');
    const shouldUseRender = apiUrl !== 'https://rag-bpql.onrender.com';
    const hasError = isProd && (isLocalhost || shouldUseRender);
    
    setConfigDetails({
      apiUrl,
      environment: process.env.NODE_ENV || 'unknown',
      isCorrect: !hasError
    });
    
    if (hasError) {
      console.error(
        'CONFIGURATION ERROR: Using incorrect API URL in production',
        {
          currentUrl: apiUrl,
          expectedUrl: 'https://rag-bpql.onrender.com',
          environment: process.env.NODE_ENV
        }
      );
      setHasConfigError(true);
    } else {
      console.log('API configuration is valid:', apiUrl);
    }
  }, []);
  
  // Don't render anything during server-side rendering
  if (typeof window === 'undefined') {
    return null;
  }
  
  if (!hasConfigError) {
    return (
      <div style={{ display: 'none' }} data-testid="config-check" data-config={JSON.stringify(configDetails)}>
        {/* Hidden element with configuration data for debugging */}
      </div>
    );
  }
  
  // Display warning banner for config errors
  return (
    <div style={{
      backgroundColor: '#ff5252',
      color: 'white',
      padding: '10px',
      textAlign: 'center',
      fontWeight: 'bold',
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      zIndex: 9999,
    }}>
      ⚠️ Configuration Error: API URL is set incorrectly. Expected https://rag-bpql.onrender.com but got {configDetails.apiUrl}
    </div>
  );
} 