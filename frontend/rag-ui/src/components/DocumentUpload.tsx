'use client';

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { ragApi } from '../api/ragApi';

/**
 * Document upload component
 */
const DocumentUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState('');
  const [author, setAuthor] = useState('');
  const [description, setDescription] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' | 'info' } | null>(null);

  // Handle file drop
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      // Only accept PDFs
      const selectedFile = acceptedFiles[0];
      if (selectedFile.type === 'application/pdf') {
        setFile(selectedFile);
        setMessage(null);
        
        // Show a warning for large files
        const fileSizeMB = selectedFile.size / (1024 * 1024);
        if (fileSizeMB > 5) {
          setMessage({
            text: `This is a large file (${fileSizeMB.toFixed(2)} MB). Processing may take a few minutes.`,
            type: 'info',
          });
        }
      } else {
        setMessage({
          text: 'Only PDF files are supported',
          type: 'error',
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
    maxFiles: 1,
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

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) {
      setMessage({
        text: 'Please select a file to upload',
        type: 'error',
      });
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    setMessage({
      text: 'Uploading and processing document. This may take a few minutes for larger files...',
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
      
      const response = await ragApi.uploadDocument(file, {
        title: title || undefined,
        author: author || undefined,
        description: description || undefined,
      });

      clearTimeout(timeoutCheck);
      setUploadProgress(100);
      
      setMessage({
        text: response.message,
        type: 'success',
      });

      // Reset form
      setFile(null);
      setTitle('');
      setAuthor('');
      setDescription('');
    } catch (error: any) {
      console.error('Upload error:', error);
      
      // Enhanced error logging
      if (error.response) {
        // The request was made and the server responded with a status code
        console.error('Server response:', error.response.data);
        console.error('Status code:', error.response.status);
        console.error('Headers:', error.response.headers);
        
        setMessage({
          text: `Server error: ${error.response.status} - ${JSON.stringify(error.response.data)}`,
          type: 'error',
        });
      } else if (error.request) {
        // The request was made but no response was received
        console.error('No response received:', error.request);
        setMessage({
          text: 'No response received from server. The document may be too large or server is busy. Try a smaller file or try again later.',
          type: 'error',
        });
      } else {
        // Something happened in setting up the request
        setMessage({
          text: `Error: ${error.message || 'Failed to upload document'}`,
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
          {file ? (
            <div>
              <p className="font-medium">{file.name}</p>
              <p className="text-sm text-gray-500">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
              {!isUploading && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setFile(null);
                  }}
                  className="mt-2 text-red-600 hover:text-red-800 text-sm"
                >
                  Remove
                </button>
              )}
            </div>
          ) : (
            <div>
              <p className="text-gray-600">
                Drag and drop a PDF file here, or click to select a file
              </p>
              <p className="text-sm text-gray-500 mt-1">Only PDF files are supported</p>
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
                ? `Processing document: ${uploadProgress}%` 
                : 'Processing complete!'}
            </p>
          </div>
        )}

        {/* Metadata fields */}
        <div className="space-y-4">
          <div>
            <label htmlFor="title" className="block text-sm font-medium text-gray-700 mb-1">
              Title (optional)
            </label>
            <input
              type="text"
              id="title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
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
            disabled={!file || isUploading}
            className={`w-full py-2 px-4 rounded-md text-white font-medium ${
              !file || isUploading
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {isUploading ? 'Uploading...' : 'Upload Document'}
          </button>
          {file && !isUploading && (
            <p className="text-xs text-gray-500 mt-1 text-center">
              Processing time depends on document size. Large documents may take several minutes.
            </p>
          )}
        </div>
      </form>
    </div>
  );
};

export default DocumentUpload; 