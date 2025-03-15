'use client';

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { ragApi } from '../api/ragApi';

/**
 * Document upload component that supports both single and multiple file uploads
 */
const DocumentUpload: React.FC = () => {
  // Define file size constants
  const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
  const MAX_BATCH_SIZE = 20 * 1024 * 1024; // 20MB total
  
  const [files, setFiles] = useState<File[]>([]);
  const [titlePrefix, setTitlePrefix] = useState('');
  const [author, setAuthor] = useState('');
  const [description, setDescription] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [message, setMessage] = useState<{ text: string; type: 'success' | 'error' | 'info' | 'warning' } | null>(null);
  
  // Define a more specific type for batch results
  interface BatchUploadResult {
    id: string;
    filename: string;
    status: 'success' | 'error';
    message?: string;
    details?: {
      chunk_count: number;
      [key: string]: number | string | boolean | object | null | undefined;
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

      // Check file size limits
      const oversizedFiles = pdfFiles.filter(file => file.size > MAX_FILE_SIZE);
      if (oversizedFiles.length > 0) {
        setMessage({
          text: `${oversizedFiles.length} files exceed the maximum size of ${(MAX_FILE_SIZE / (1024 * 1024)).toFixed(1)}MB: ${oversizedFiles.map(f => f.name).join(', ')}`,
          type: 'error',
        });
        
        // Filter out oversized files if there are any valid files
        const validFiles = pdfFiles.filter(file => file.size <= MAX_FILE_SIZE);
        if (validFiles.length > 0) {
          setFiles(validFiles);
          setMessage({
            text: `${oversizedFiles.length} files were removed because they exceed the ${(MAX_FILE_SIZE / (1024 * 1024)).toFixed(1)}MB limit. Proceeding with ${validFiles.length} valid files.`,
            type: 'warning',
          });
        }
        return;
      }
      
      // Check total batch size
      const totalSize = pdfFiles.reduce((acc, file) => acc + file.size, 0);
      if (totalSize > MAX_BATCH_SIZE) {
        setMessage({
          text: `Total upload size (${(totalSize / (1024 * 1024)).toFixed(2)}MB) exceeds the maximum of ${(MAX_BATCH_SIZE / (1024 * 1024))}MB.`,
          type: 'error',
        });
        return;
      }
      
      setFiles(pdfFiles);
      setMessage(null);
      
      // Calculate total size
      const totalSizeMB = totalSize / (1024 * 1024);
      
      // Show a warning for large uploads
      if (totalSizeMB > 2) {
        setMessage({
          text: `This is a large upload (${totalSizeMB.toFixed(2)}MB total). Processing may take several minutes.`,
          type: 'info',
        });
      }
    }
  }, [MAX_BATCH_SIZE, MAX_FILE_SIZE]);

  // Configure dropzone
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    maxFiles: 10,
    multiple: true,
  });

  // Track upload progress (for actual uploads, no longer simulated)
  const updateUploadProgress = (progress: number) => {
    setUploadProgress(progress);
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
    setMessage(null);
    setBatchResults(null);

    try {
      if (files.length === 1) {
        // Single file upload
        const file = files[0];
        const metadata = {
          title: titlePrefix ? `${titlePrefix} - ${file.name}` : file.name,
          author,
          description,
        };

        // Use the onProgress callback to update progress
        await ragApi.uploadDocument(file, metadata, updateUploadProgress);

        // Upload successful
        setMessage({
          text: 'Document uploaded successfully!',
          type: 'success',
        });
      } else {
        // Multiple files upload
        const metadata = {
          titlePrefix,
          author,
          description,
        };

        // Use the onProgress callback to update progress
        const result = await ragApi.uploadMultipleDocuments(files, metadata, updateUploadProgress);

        // Show batch results
        setMessage({
          text: `Upload complete! Successfully processed ${result.successful_count} out of ${result.successful_count + result.failed_count} documents.`,
          type: result.failed_count === 0 ? 'success' : 'warning',
        });
        setBatchResults(result.results);
      }
    } catch (error) {
      // Handle upload errors
      console.error('Upload error:', error);
      setMessage({
        text: `Error uploading document(s): ${error instanceof Error ? error.message : 'Unknown error'}`,
        type: 'error',
      });
    } finally {
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
                : message.type === 'info'
                  ? 'bg-blue-100 text-blue-700'
                  : 'bg-yellow-100 text-yellow-700'
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

        {/* Show upload progress bar */}
        {isUploading && (
          <div className="mb-6">
            <div className="w-full bg-gray-200 rounded-full h-4 mb-2">
              <div
                className="bg-blue-600 h-4 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
            <div className="text-sm text-gray-600">
              {uploadProgress < 50 ? (
                <span>Uploading document... {uploadProgress}%</span>
              ) : uploadProgress < 95 ? (
                <span>Processing document... {uploadProgress}%</span>
              ) : uploadProgress < 100 ? (
                <span>Almost done... {uploadProgress}%</span>
              ) : (
                <span>Upload complete! 100%</span>
              )}
            </div>
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

      {/* Add a note about processing time */}
      <div className="text-sm text-gray-500 mt-6">
        <p><strong>Note:</strong> The upload process consists of two phases:</p>
        <ol className="list-decimal ml-5 mt-2">
          <li>File transfer to the server (typically quick)</li>
          <li>Document processing, embedding, and storage (may take several minutes for large documents)</li>
        </ol>
        <p className="mt-2">Please wait for the entire process to complete before querying your documents.</p>
      </div>
    </div>
  );
};

export default DocumentUpload; 