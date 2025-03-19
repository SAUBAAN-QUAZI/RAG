import { useEffect, useState } from 'react';
import config from '../config';

/**
 * Component to verify configuration at runtime
 * Displays warnings if using incorrect API URL in production
 */
export default function ConfigCheck() {
  const [hasConfigError, setHasConfigError] = useState(false);
  
  useEffect(() => {
    // Check if we're in production and using localhost
    const isProd = process.env.NODE_ENV === 'production';
    const isLocalhost = config.apiUrl.includes('localhost');
    const shouldUseRender = config.apiUrl !== 'https://rag-bpql.onrender.com';
    
    if (isProd && (isLocalhost || shouldUseRender)) {
      console.error(
        'CONFIGURATION ERROR: Using incorrect API URL in production',
        {
          currentUrl: config.apiUrl,
          expectedUrl: 'https://rag-bpql.onrender.com',
          environment: process.env.NODE_ENV
        }
      );
      setHasConfigError(true);
    } else {
      console.log('API configuration is valid:', config.apiUrl);
    }
  }, []);
  
  if (!hasConfigError) {
    return null;
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
      ⚠️ Configuration Error: API URL is set incorrectly. Please contact the administrator.
    </div>
  );
} 