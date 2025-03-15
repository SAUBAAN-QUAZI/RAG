'use client';

import React from 'react';

interface ErrorDisplayProps {
  error: string | null;
  onRetry?: () => void;
  retryCount?: number;
  isRetrying?: boolean;
  maxRetries?: number;
}

/**
 * Reusable component for displaying and handling API errors
 */
const ErrorDisplay: React.FC<ErrorDisplayProps> = ({
  error,
  onRetry,
  retryCount = 0,
  isRetrying = false,
  maxRetries = 3
}) => {
  if (!error) return null;

  // Format error message to handle line breaks
  const formattedError = error.replace(/\n/g, '<br>');
  
  // Determine if we should show retry button
  const showRetry = !!onRetry;
  
  // Check if we've reached max retries
  const reachedMaxRetries = retryCount >= maxRetries;

  return (
    <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-md mb-4 flex flex-col">
      <div className="font-medium mb-2">Error</div>
      <div dangerouslySetInnerHTML={{ __html: formattedError }} />
      
      {showRetry && (
        <div className="mt-3 flex items-center gap-3">
          <button
            onClick={onRetry}
            disabled={isRetrying || reachedMaxRetries}
            className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-md text-sm font-medium disabled:opacity-50"
          >
            {isRetrying ? 'Retrying...' : `Retry${retryCount > 0 ? ` (${retryCount})` : ''}`}
          </button>
          
          {retryCount > 0 && !reachedMaxRetries && (
            <span className="text-sm text-red-600">
              {maxRetries - retryCount} retries remaining
            </span>
          )}
        </div>
      )}
      
      {reachedMaxRetries && showRetry && (
        <div className="mt-2 text-sm bg-red-100 p-2 rounded-md">
          <strong>Maximum retry attempts reached.</strong> The server might be experiencing issues or your request may be too complex.
          Try again later or contact support if the issue persists.
        </div>
      )}
      
      {/* Show reconnection tips for network errors */}
      {error.includes('Network error') && (
        <div className="mt-4 text-sm bg-red-100 p-3 rounded-md">
          <strong>Connection Troubleshooting:</strong>
          <ul className="list-disc pl-5 mt-2 space-y-1">
            <li>Verify the backend server is running</li>
            <li>Check your network connection</li>
            <li>Ensure the API URL is correctly configured</li>
            <li>Check for CORS errors in browser console</li>
          </ul>
        </div>
      )}
    </div>
  );
};

export default ErrorDisplay; 