'use client';

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { ragApi } from '../api/ragApi';

/**
 * Document upload component that supports both single and multiple file uploads
 */
const DocumentUpload: React.FC = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [titlePrefix, setTitlePrefix] = useState('');
  const [author, setAuthor] = useState('');
  const [description, setDescription] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' | 'info' } | null>(null);
  
  // Define a more specific type for batch results
  interface BatchUploadResult {
    id: string;
    filename: string;
    status: 'success' | 'error';
    message?: string;
    details?: {
      chunk_count: number;
      [key: string]: any; // Include any other properties that might be in details
    };
  }
  
  const [batchResults, setBatchResults] = useState<BatchUploadResult[] | null>(null);

  // Handle file drop
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      // Filter to only PDFs
      const pdfFiles = acceptedFiles.filter(file => file.type === 'application/pdf');
      
      if (pdfFiles.length === 0) {
        setMessage({
          text: 'Only PDF files are supported',
          type: 'error',
        });
        return;
      }
      
      if (pdfFiles.length > 10) {
        setMessage({
          text: 'Maximum 10 files can be uploaded at once',
          type: 'error',
        });
        return;
      }
      
      setFiles(pdfFiles);
      setMessage(null);
      
      // Calculate total size
      const totalSizeMB = pdfFiles.reduce((acc, file) => acc + file.size, 0) / (1024 * 1024);
      
      // Show a warning for large uploads
      if (totalSizeMB > 5) {
        setMessage({
          text: `This is a large upload (${totalSizeMB.toFixed(2)} MB total). Processing may take several minutes.`,
          type: 'info',
        });
      }
    }
  }, []);

  // Configure dropzone
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    maxFiles: 10,
    multiple: true,
  });

  // Progress tracking function
  const trackProgress = () => {
    // Simulated progress for processing - actual upload progress is handled by axios
    // This just gives user feedback that something is happening during server processing
    let progress = 0;
    const interval = setInterval(() => {
      progress += 1;
      if (progress >= 95) {
        clearInterval(interval);
        return;
      }
      setUploadProgress(progress);
    }, 1000);
    
    return () => clearInterval(interval);
  };

  // Remove a file from the list (currently unused but kept for future use)
  // Commented out to satisfy ESLint while preserving for future functionality
  /*
  const removeFile = (indexToRemove: number) => {
    setFiles(prevFiles => prevFiles.filter((_, index) => index !== indexToRemove));
  };
  */

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (files.length === 0) {
      setMessage({
        text: 'Please select at least one file to upload',
        type: 'error',
      });
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    setBatchResults(null);
    setMessage({
      text: `Uploading and processing ${files.length} document${files.length > 1 ? 's' : ''}. This may take a few minutes...`,
      type: 'info',
    });
    
    // Start progress tracking
    const stopTracking = trackProgress();

    try {
      // Set up a timeout to check server status if it takes too long
      const timeoutCheck = setTimeout(() => {
        if (isUploading) {
          setMessage({
            text: 'Still processing... For large documents, this can take several minutes.',
            type: 'info',
          });
        }
      }, 60000); // Show extra message after 1 minute
      
      // Use batch upload if multiple files, otherwise use single upload
      let response;
      if (files.length > 1) {
        response = await ragApi.uploadMultipleDocuments(files, {
          titlePrefix,
          author: author || undefined,
          description: description || undefined,
        });
        
        // Store detailed results for display
        if (response.results) {
          setBatchResults(response.results);
        }
      } else {
        response = await ragApi.uploadDocument(files[0], {
          title: titlePrefix ? `${titlePrefix} - ${files[0].name}` : undefined,
          author: author || undefined,
          description: description || undefined,
        });
      }

      clearTimeout(timeoutCheck);
      setUploadProgress(100);
      
      setMessage({
        text: response.message,
        type: 'success',
      });

      // Reset form on success
      setFiles([]);
      // Don't reset metadata fields to allow for continued uploads with same metadata
    } catch (error) {
      console.error('Upload error:', error);
      
      // Define proper error types
      interface ServerError {
        response?: {
          data: unknown;
          status: number;
          headers: unknown;
        };
        request?: unknown;
        message?: string;
      }
      
      // Enhanced error logging
      const serverError = error as ServerError;
      
      if (serverError.response) {
        // The request was made and the server responded with a status code
        console.error('Server response:', serverError.response.data);
        console.error('Status code:', serverError.response.status);
        
        setMessage({
          text: `Server error: ${serverError.response.status} - ${JSON.stringify(serverError.response.data)}`,
          type: 'error',
        });
      } else if (serverError.request) {
        // The request was made but no response was received
        console.error('No response received:', serverError.request);
        
        setMessage({
          text: 'No response received from server. It might be down or unreachable.',
          type: 'error',
        });
      } else {
        // Something happened in setting up the request that triggered an error
        console.error('Error message:', serverError.message);
        
        setMessage({
          text: `Error: ${serverError.message}`,
          type: 'error',
        });
      }
    } finally {
      stopTracking();
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Upload Document</h1>

      {/* Message display */}
      {message && (
        <div
          className={`p-4 mb-6 rounded-md ${
            message.type === 'success' 
              ? 'bg-green-100 text-green-700' 
              : message.type === 'error'
                ? 'bg-red-100 text-red-700'
                : 'bg-blue-100 text-blue-700'
          }`}
        >
          {message.text}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* File dropzone */}
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-md p-6 text-center cursor-pointer transition-colors ${
            isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'
          } ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <input {...getInputProps()} disabled={isUploading} />
          {files.length > 0 ? (
            <div>
              <p className="font-medium">{files.map(file => file.name).join(', ')}</p>
              <p className="text-sm text-gray-500">
                {(files.reduce((acc, file) => acc + file.size, 0) / 1024 / 1024).toFixed(2)} MB
              </p>
              {!isUploading && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setFiles([]);
                  }}
                  className="mt-2 text-red-600 hover:text-red-800 text-sm"
                >
                  Remove all files
                </button>
              )}
            </div>
          ) : (
            <div>
              <p className="text-gray-600">
                Drag and drop PDF files here, or click to select files
              </p>
              <p className="text-sm text-gray-500 mt-1">Maximum 10 files, each up to 5 MB</p>
            </div>
          )}
        </div>

        {/* Progress bar */}
        {isUploading && (
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className="bg-blue-600 h-2.5 rounded-full transition-all duration-300" 
              style={{ width: `${uploadProgress}%` }}
            ></div>
            <p className="text-sm text-gray-500 mt-1 text-center">
              {uploadProgress < 100 
                ? `Processing documents: ${uploadProgress}%` 
                : 'Processing complete!'}
            </p>
          </div>
        )}

        {/* Metadata fields */}
        <div className="space-y-4">
          <div>
            <label htmlFor="titlePrefix" className="block text-sm font-medium text-gray-700 mb-1">
              Title prefix (optional)
            </label>
            <input
              type="text"
              id="titlePrefix"
              value={titlePrefix}
              onChange={(e) => setTitlePrefix(e.target.value)}
              disabled={isUploading}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label htmlFor="author" className="block text-sm font-medium text-gray-700 mb-1">
              Author (optional)
            </label>
            <input
              type="text"
              id="author"
              value={author}
              onChange={(e) => setAuthor(e.target.value)}
              disabled={isUploading}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-1">
              Description (optional)
            </label>
            <textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
              disabled={isUploading}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* Submit button */}
        <div>
          <button
            type="submit"
            disabled={files.length === 0 || isUploading}
            className={`w-full py-2 px-4 rounded-md text-white font-medium ${
              files.length === 0 || isUploading
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {isUploading ? 'Uploading...' : 'Upload Documents'}
          </button>
          {files.length > 0 && !isUploading && (
            <p className="text-xs text-gray-500 mt-1 text-center">
              Processing time depends on document size. Large documents may take several minutes.
            </p>
          )}
        </div>
      </form>

      {/* Display batch results if available */}
      {batchResults && batchResults.length > 0 && (
        <div className="mt-8">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Upload Results</h3>
          <div className="bg-white overflow-hidden shadow rounded-lg divide-y divide-gray-200">
            <div className="px-4 py-4 sm:px-6 font-medium flex justify-between">
              <div>Filename</div>
              <div>Status</div>
            </div>
            <div className="divide-y divide-gray-200">
              {batchResults.map((result, index) => (
                <div key={index} className="px-4 py-4 sm:px-6 flex justify-between items-center">
                  <div className="truncate max-w-xs">{result.filename}</div>
                  <div>
                    {result.status === 'success' ? (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                        Success {result.details && `(${result.details.chunk_count} chunks)`}
                      </span>
                    ) : (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                        Failed
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Instructions section */}
      <div className="mt-8 bg-blue-50 p-4 rounded-md">
        <h3 className="text-lg font-medium text-blue-900 mb-2">About Document Upload</h3>
        <ul className="list-disc list-inside text-sm text-blue-800 space-y-1">
          <li>Supported formats: PDF files only</li>
          <li>Maximum file size: 50MB per file</li>
          <li>You can upload up to 10 files at once</li>
          <li>Processing may take several minutes for large documents</li>
          <li>Documents will be chunked and embedded for retrieval</li>
          <li>After uploading, you can query the documents in the Chat tab</li>
        </ul>
      </div>
    </div>
  );
};

export default DocumentUpload; 