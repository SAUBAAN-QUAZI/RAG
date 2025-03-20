'use client';

import React, { useState, useRef, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Input,
  Text,
  VStack,
  HStack,
  Progress,
  useToast,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  Tooltip,
  IconButton,
  Flex,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  useDisclosure,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalCloseButton,
  ModalBody,
  ModalFooter,
  Spinner,
} from '@chakra-ui/react';
import { ragApi, DocumentUploadResult, BatchUploadResultItem } from '../api/ragApi';
import { FaUpload, FaTrash, FaFile, FaInfoCircle } from 'react-icons/fa';

// Constants for upload limits
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50 MB max file size
const MAX_BATCH_SIZE = 5; // Maximum number of files in a batch

/**
 * Interface for document metadata
 */
interface DocumentMetadata {
  title?: string;
  author?: string;
  description?: string;
}

/**
 * Interface for batch upload results
 */
interface BatchUploadResult {
  id: string;
  filename: string;
  status: 'success' | 'error' | 'processing';
  message?: string;
  ragie_document_id?: string; // Added for Ragie integration
  details?: any;
}

/**
 * DocumentUpload component for handling document uploads to the RAG system
 */
const DocumentUpload: React.FC = () => {
  // State for upload form
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [metadata, setMetadata] = useState<DocumentMetadata>({});
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [batchResults, setBatchResults] = useState<BatchUploadResult[]>([]);
  const [showResult, setShowResult] = useState(false);
  
  // For document deletion
  const [isDeletingDocument, setIsDeletingDocument] = useState(false);
  const [documentToDelete, setDocumentToDelete] = useState<string | null>(null);
  
  // State for document list
  const [documents, setDocuments] = useState<Array<{ id: string; name: string; status: string; metadata: Record<string, any> }>>([]);
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(false);
  
  // Access Chakra UI toast
  const toast = useToast();
  
  // Modal control for document details
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [selectedDocument, setSelectedDocument] = useState<any>(null);
  
  /**
   * Function to fetch all documents
   */
  const fetchDocuments = useCallback(async () => {
    setIsLoadingDocuments(true);
    try {
      const docs = await ragApi.listDocuments();
      setDocuments(docs);
    } catch (error) {
      console.error('Error fetching documents:', error);
      toast({
        title: 'Error fetching documents',
        description: error instanceof Error ? error.message : 'Failed to fetch documents',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoadingDocuments(false);
    }
  }, [toast]);
  
  // Fetch documents on component mount
  React.useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);
  
  /**
   * Handle file drop using react-dropzone
   */
  const onDrop = useCallback((acceptedFiles: File[]) => {
    // Filter for PDF files only
    const pdfFiles = acceptedFiles.filter(
      file => file.type === 'application/pdf' ||
              file.name.toLowerCase().endsWith('.pdf')
    );
    
    if (pdfFiles.length < acceptedFiles.length) {
      toast({
        title: 'Non-PDF files excluded',
        description: 'Only PDF files are supported at this time.',
        status: 'warning',
        duration: 5000,
        isClosable: true,
      });
    }
    
    // Apply size limit filter
    const validFiles = pdfFiles.filter(file => {
      if (file.size > MAX_FILE_SIZE) {
        toast({
          title: 'File too large',
          description: `"${file.name}" exceeds the maximum file size of 50 MB.`,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
        return false;
      }
      return true;
    });
    
    // Limit the number of files
    if (validFiles.length > MAX_BATCH_SIZE) {
      toast({
        title: 'Too many files',
        description: `Only ${MAX_BATCH_SIZE} files can be uploaded at once. The first ${MAX_BATCH_SIZE} valid files were selected.`,
        status: 'warning',
        duration: 5000,
        isClosable: true,
      });
      setSelectedFiles(validFiles.slice(0, MAX_BATCH_SIZE));
    } else {
      setSelectedFiles(validFiles);
    }
  }, [toast]);
  
  /**
   * Configure dropzone
   */
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    maxFiles: MAX_BATCH_SIZE,
    maxSize: MAX_FILE_SIZE,
  });
  
  /**
   * Handle input change for metadata fields
   */
  const handleMetadataChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setMetadata(prev => ({
      ...prev,
      [name]: value,
    }));
  };
  
  /**
   * Handle upload submit
   */
  const handleUpload = async () => {
    if (selectedFiles.length === 0) {
      toast({
        title: 'No files selected',
        description: 'Please select one or more PDF files to upload.',
        status: 'warning',
        duration: 5000,
        isClosable: true,
      });
      return;
    }
    
    setIsUploading(true);
    setUploadProgress(0);
    setBatchResults([]);
    setShowResult(false);
    
    // Log upload attempt for debugging
    console.log('Starting document upload:', {
      fileCount: selectedFiles.length,
      files: selectedFiles.map(f => ({
        name: f.name,
        size: `${(f.size / 1024 / 1024).toFixed(2)} MB`,
        type: f.type
      }))
    });
    
    try {
      if (selectedFiles.length === 1) {
        // Single file upload
        toast({
          title: 'Uploading document',
          description: `Uploading ${selectedFiles[0].name} (${(selectedFiles[0].size / 1024 / 1024).toFixed(2)} MB)`,
          status: 'info',
          duration: 5000,
          isClosable: true,
        });
        
        const result: DocumentUploadResult = await ragApi.uploadDocument(
          selectedFiles[0],
          metadata,
          (progress) => setUploadProgress(progress)
        );
        
        // Create a standardized result for UI
        setBatchResults([{
          id: result.document_id || 'unknown',
          filename: selectedFiles[0].name,
          status: 'success',
          message: result.message,
          ragie_document_id: result.ragie_document_id,
        }]);
        
        toast({
          title: 'Upload successful',
          description: `Document ${selectedFiles[0].name} was uploaded successfully.`,
          status: 'success',
          duration: 5000,
          isClosable: true,
        });
      } else {
        // Batch upload
        toast({
          title: 'Uploading multiple documents',
          description: `Uploading ${selectedFiles.length} documents.`,
          status: 'info',
          duration: 5000,
          isClosable: true,
        });
        
        const batchMetadata = {
          titlePrefix: metadata.title,
          author: metadata.author,
          description: metadata.description,
        };
        
        const batchResult = await ragApi.uploadMultipleDocuments(
          selectedFiles,
          batchMetadata,
          (progress) => setUploadProgress(progress)
        );
        
        setBatchResults(batchResult.results || []);
        
        toast({
          title: 'Batch upload completed',
          description: `${batchResult.successful_count} documents processed successfully, ${batchResult.failed_count} failed.`,
          status: batchResult.failed_count === 0 ? 'success' : 'warning',
          duration: 5000,
          isClosable: true,
        });
      }
      
      setShowResult(true);
      
      // Refresh document list after successful upload
      fetchDocuments();
    } catch (error: any) {
      console.error('Upload error:', error);
      
      // Reset progress
      setUploadProgress(0);
      
      // Handle specific error cases with user-friendly messages
      if (error.message && error.message.includes('405 Method Not Allowed')) {
        toast({
          title: 'Upload API Error',
          description: 'The server does not support document uploads at this endpoint. Please check server configuration.',
          status: 'error',
          duration: 10000,
          isClosable: true,
        });
      } 
      else if (error.message && error.message.includes('timeout')) {
        toast({
          title: 'Upload Timeout',
          description: 'The upload took too long and timed out. Try a smaller document or check your connection.',
          status: 'error',
          duration: 10000,
          isClosable: true,
        });
      }
      else if (error.message && error.message.includes('Network error')) {
        toast({
          title: 'Network Error',
          description: 'Cannot connect to the server. Please check your internet connection and try again.',
          status: 'error',
          duration: 10000,
          isClosable: true,
        });
      }
      else if (error.message && error.message.includes('413')) {
        toast({
          title: 'File Too Large',
          description: 'The server rejected the file because it is too large. Please try a smaller file.',
          status: 'error',
          duration: 10000,
          isClosable: true,
        });
      }
      else {
        toast({
          title: 'Upload Failed',
          description: error.message || 'An unexpected error occurred during upload.',
          status: 'error',
          duration: 10000,
          isClosable: true,
        });
      }
      
      // Optionally, set an error result to display in the UI
      if (selectedFiles.length === 1) {
        setBatchResults([{
          id: 'error',
          filename: selectedFiles[0].name,
          status: 'error',
          message: error.message || 'Upload failed',
        }]);
        setShowResult(true);
      }
    } finally {
      setIsUploading(false);
    }
  };
  
  /**
   * Clear selected files
   */
  const handleClear = () => {
    setSelectedFiles([]);
    setBatchResults([]);
    setShowResult(false);
  };
  
  /**
   * Handle document deletion
   */
  const handleDeleteDocument = async (documentId: string) => {
    setDocumentToDelete(documentId);
    setIsDeletingDocument(true);
    
    try {
      await ragApi.deleteDocument(documentId);
      toast({
        title: 'Document deleted',
        description: 'The document has been successfully deleted.',
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
      
      // Refresh the documents list
      fetchDocuments();
    } catch (error) {
      console.error('Error deleting document:', error);
      toast({
        title: 'Deletion failed',
        description: error instanceof Error ? error.message : 'Failed to delete document',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsDeletingDocument(false);
      setDocumentToDelete(null);
    }
  };
  
  /**
   * Show document details in modal
   */
  const handleShowDetails = (document: any) => {
    setSelectedDocument(document);
    onOpen();
  };
  
  /**
   * Get badge color based on document status
   */
  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'success':
      case 'complete':
      case 'indexed':
        return 'green';
      case 'processing':
        return 'yellow';
      case 'error':
      case 'failed':
        return 'red';
      default:
        return 'gray';
    }
  };
  
  return (
    <Box p={4} borderWidth="1px" borderRadius="lg" bg="white" shadow="md">
      <VStack spacing={6} align="stretch">
        <Flex justify="space-between" align="center">
          <Text fontSize="2xl" fontWeight="bold">Document Upload</Text>
          <Button 
            leftIcon={<FaUpload />} 
            colorScheme="blue" 
            size="sm" 
            onClick={fetchDocuments} 
            isLoading={isLoadingDocuments}
          >
            Refresh
          </Button>
        </Flex>
        
        {/* Upload Form */}
        <Box borderWidth="1px" borderRadius="md" p={4}>
          <VStack spacing={4} align="stretch">
            <Box
              {...getRootProps()}
              p={5}
              borderWidth="2px"
              borderRadius="md"
              borderStyle="dashed"
              borderColor={isDragActive ? "blue.400" : "gray.300"}
              bg={isDragActive ? "blue.50" : "gray.50"}
              cursor="pointer"
              _hover={{ borderColor: "blue.300", bg: "blue.50" }}
              transition="all 0.2s"
            >
              <input {...getInputProps()} />
              <VStack spacing={2} justify="center">
                <Text textAlign="center" fontSize="sm" color="gray.600">
                  {isDragActive
                    ? "Drop the PDF files here..."
                    : "Drag and drop PDF files here, or click to select files"}
                </Text>
                <Text textAlign="center" fontSize="xs" color="gray.500">
                  Maximum file size: 50 MB, up to 5 files at once
                </Text>
                <Box>
                  <Button size="sm" leftIcon={<FaFile />} colorScheme="blue" variant="outline">
                    Select Files
                  </Button>
                </Box>
              </VStack>
            </Box>
            
            {/* Selected Files List */}
            {selectedFiles.length > 0 && (
              <Box>
                <Text fontSize="sm" fontWeight="medium" mb={2}>
                  Selected Files ({selectedFiles.length}):
                </Text>
                <VStack spacing={1} align="stretch">
                  {selectedFiles.map((file, index) => (
                    <Flex 
                      key={index} 
                      p={1} 
                      borderWidth="1px" 
                      borderRadius="md"
                      justify="space-between"
                      align="center"
                    >
                      <HStack>
                        <FaFile size="0.8em" />
                        <Text fontSize="sm" isTruncated maxW="300px">
                          {file.name}
                        </Text>
                      </HStack>
                      <Text fontSize="xs" color="gray.500">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </Text>
                    </Flex>
                  ))}
                </VStack>
              </Box>
            )}
            
            {/* Metadata Form */}
            <FormControl>
              <FormLabel fontSize="sm">Title or Title Prefix (for batch uploads)</FormLabel>
              <Input
                size="sm"
                name="title"
                value={metadata.title || ''}
                onChange={handleMetadataChange}
                placeholder="Enter a title for the document(s)"
              />
            </FormControl>
            <FormControl>
              <FormLabel fontSize="sm">Author (optional)</FormLabel>
              <Input
                size="sm"
                name="author"
                value={metadata.author || ''}
                onChange={handleMetadataChange}
                placeholder="Enter author name"
              />
            </FormControl>
            <FormControl>
              <FormLabel fontSize="sm">Description (optional)</FormLabel>
              <Input
                size="sm"
                name="description"
                value={metadata.description || ''}
                onChange={handleMetadataChange}
                placeholder="Enter a brief description"
              />
            </FormControl>
            
            {/* Upload Progress */}
            {isUploading && (
              <Box>
                <Text fontSize="sm" mb={1}>
                  Uploading and processing files... {uploadProgress}%
                </Text>
                <Progress 
                  value={uploadProgress} 
                  size="sm" 
                  colorScheme="blue" 
                  isAnimated
                  hasStripe
                />
              </Box>
            )}
            
            {/* Action Buttons */}
            <HStack spacing={4} justify="flex-end">
              <Button
                size="sm"
                onClick={handleClear}
                isDisabled={selectedFiles.length === 0 || isUploading}
              >
                Clear
              </Button>
              <Button
                size="sm"
                colorScheme="blue"
                onClick={handleUpload}
                isLoading={isUploading}
                loadingText="Uploading..."
                leftIcon={<FaUpload />}
                isDisabled={selectedFiles.length === 0 || isUploading}
              >
                Upload
              </Button>
            </HStack>
          </VStack>
        </Box>
        
        {/* Upload Results */}
        {showResult && batchResults.length > 0 && (
          <Box borderWidth="1px" borderRadius="md" p={4}>
            <Text fontSize="md" fontWeight="bold" mb={2}>
              Upload Results
            </Text>
            <Table size="sm" variant="simple">
              <Thead>
                <Tr>
                  <Th>Filename</Th>
                  <Th>Status</Th>
                  <Th>Message</Th>
                </Tr>
              </Thead>
              <Tbody>
                {batchResults.map((result, index) => (
                  <Tr key={index}>
                    <Td>{result.filename}</Td>
                    <Td>
                      <Badge colorScheme={getStatusColor(result.status)}>
                        {result.status}
                      </Badge>
                    </Td>
                    <Td fontSize="xs">{result.message}</Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          </Box>
        )}
        
        {/* Documents List */}
        <Box borderWidth="1px" borderRadius="md" p={4}>
          <Text fontSize="md" fontWeight="bold" mb={2}>
            Uploaded Documents
          </Text>
          
          {isLoadingDocuments ? (
            <Flex justify="center" p={4}>
              <Spinner size="md" />
            </Flex>
          ) : documents.length > 0 ? (
            <Table size="sm" variant="simple">
              <Thead>
                <Tr>
                  <Th>Name</Th>
                  <Th>Status</Th>
                  <Th>Actions</Th>
                </Tr>
              </Thead>
              <Tbody>
                {documents.map((doc) => (
                  <Tr key={doc.id}>
                    <Td isTruncated maxW="200px">{doc.name}</Td>
                    <Td>
                      <Badge colorScheme={getStatusColor(doc.status)}>
                        {doc.status}
                      </Badge>
                    </Td>
                    <Td>
                      <HStack spacing={2}>
                        <Tooltip label="View Details">
                          <IconButton
                            aria-label="View document details"
                            icon={<FaInfoCircle />}
                            size="xs"
                            onClick={() => handleShowDetails(doc)}
                          />
                        </Tooltip>
                        <Tooltip label="Delete Document">
                          <IconButton
                            aria-label="Delete document"
                            icon={<FaTrash />}
                            size="xs"
                            colorScheme="red"
                            variant="ghost"
                            isLoading={isDeletingDocument && documentToDelete === doc.id}
                            onClick={() => handleDeleteDocument(doc.id)}
                          />
                        </Tooltip>
                      </HStack>
                    </Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          ) : (
            <Text textAlign="center" fontSize="sm" color="gray.500" p={4}>
              No documents found. Upload some documents to get started.
            </Text>
          )}
        </Box>
      </VStack>
      
      {/* Document Details Modal */}
      <Modal isOpen={isOpen} onClose={onClose} size="lg">
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Document Details</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            {selectedDocument && (
              <VStack align="stretch" spacing={4}>
                <Box>
                  <Text fontWeight="bold">Name:</Text>
                  <Text>{selectedDocument.name}</Text>
                </Box>
                <Box>
                  <Text fontWeight="bold">ID:</Text>
                  <Text fontSize="sm" fontFamily="monospace">{selectedDocument.id}</Text>
                </Box>
                <Box>
                  <Text fontWeight="bold">Status:</Text>
                  <Badge colorScheme={getStatusColor(selectedDocument.status)}>
                    {selectedDocument.status}
                  </Badge>
                </Box>
                
                {/* Document Metadata */}
                <Accordion allowToggle>
                  <AccordionItem>
                    <AccordionButton>
                      <Box flex="1" textAlign="left">
                        <Text fontWeight="semibold">Metadata</Text>
                      </Box>
                      <AccordionIcon />
                    </AccordionButton>
                    <AccordionPanel pb={4}>
                      <VStack align="stretch" spacing={2}>
                        {selectedDocument.metadata && Object.entries(selectedDocument.metadata).map(([key, value]) => (
                          <Box key={key}>
                            <Text fontSize="sm" fontWeight="bold">{key}:</Text>
                            <Text fontSize="sm" whiteSpace="pre-wrap" overflowWrap="break-word">
                              {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                            </Text>
                          </Box>
                        ))}
                        {(!selectedDocument.metadata || Object.keys(selectedDocument.metadata).length === 0) && (
                          <Text fontSize="sm" color="gray.500">No metadata available</Text>
                        )}
                      </VStack>
                    </AccordionPanel>
                  </AccordionItem>
                </Accordion>
              </VStack>
            )}
          </ModalBody>
          <ModalFooter>
            <Button colorScheme="blue" mr={3} onClick={onClose}>
              Close
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Box>
  );
};

export default DocumentUpload; 