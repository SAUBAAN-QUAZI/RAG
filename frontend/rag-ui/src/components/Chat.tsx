'use client';

import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Input,
  Button,
  VStack,
  HStack,
  Text,
  Flex,
  IconButton,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Divider,
  useToast,
  Card,
  CardBody,
  Badge,
  Spinner,
  Textarea,
  Select,
  Switch,
  FormControl,
  FormLabel,
  Tooltip,
  Collapse,
  useDisclosure,
  Drawer,
  DrawerBody,
  DrawerHeader,
  DrawerOverlay,
  DrawerContent,
  DrawerCloseButton,
} from '@chakra-ui/react';
import { FiSend, FiSettings, FiCopy, FiRefreshCw } from 'react-icons/fi';
import ReactMarkdown from 'react-markdown';
import { ragApi, QueryRequest, QueryResponse } from '../api/ragApi';

// Define types for chat messages
interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Array<{
    text: string;
    score: number;
    document_id: string;
    document_name?: string;
  }>;
  error?: boolean;
  timestamp: Date;
  loading?: boolean;
  noResults?: boolean;
}

// Define type for document selection
interface DocumentOption {
  id: string;
  name: string;
}

/**
 * Chat component for interacting with documents through RAG
 */
const Chat: React.FC = () => {
  // State for managing messages
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const toast = useToast();
  
  // State for query settings
  const [rerank, setRerank] = useState(true);
  const [topK, setTopK] = useState(3);
  const [showTimings, setShowTimings] = useState(false);
  const { isOpen: isSettingsOpen, onOpen: onSettingsOpen, onClose: onSettingsClose } = useDisclosure();
  
  // State for document selection
  const [documents, setDocuments] = useState<DocumentOption[]>([]);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(false);
  
  // Fetch available documents on component mount
  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        setIsLoadingDocuments(true);
        const docs = await ragApi.listDocuments();
        const documentOptions = docs.map(doc => ({
          id: doc.id,
          name: doc.name || `Document ${doc.id}`
        }));
        setDocuments(documentOptions);
      } catch (error) {
        console.error('Error fetching documents:', error);
        toast({
          title: 'Failed to load documents',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
      } finally {
        setIsLoadingDocuments(false);
      }
    };
    
    fetchDocuments();
  }, [toast]);
  
  // Scroll to bottom of messages when new message is added
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  // Handle sending a message
  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;
    
    // Generate a unique ID for the message
    const messageId = Date.now().toString();
    
    // Add user message to chat
    const userMessage: Message = {
      id: messageId,
      role: 'user',
      content: inputValue,
      timestamp: new Date(),
    };
    
    // Add temporary loading message for assistant
    const loadingMessage: Message = {
      id: `${messageId}-response`,
      role: 'assistant',
      content: '',
      loading: true,
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, userMessage, loadingMessage]);
    setInputValue('');
    setIsLoading(true);
    
    try {
      // Check if we have the document ID from Ragie in the metadata
      const isDocQuery = documents.length > 0 && (selectedDocuments.length === 0);
      let targetDocumentIds = selectedDocuments;
      
      // If no documents are specifically selected but we have documents, 
      // let's try using all available documents to increase chances of finding matches
      if (isDocQuery) {
        targetDocumentIds = documents.map(doc => doc.id);
      }
      
      // Prepare query request
      const queryRequest: QueryRequest = {
        query: inputValue,
        rerank,
        top_k: topK,
        show_timings: showTimings,
      };
      
      // Add document IDs if any are available
      if (targetDocumentIds.length > 0) {
        queryRequest.document_ids = targetDocumentIds;
      }
      
      // Send query to API
      const response = await ragApi.query(queryRequest);
      
      // Check if we got empty results
      const hasResults = response.chunks && response.chunks.length > 0;
      
      // Create a better message when no results are found
      let finalContent = response.response;
      if (!hasResults && finalContent.includes("I don't have enough information")) {
        if (targetDocumentIds.length > 0) {
          const docNames = documents
            .filter(doc => targetDocumentIds.includes(doc.id))
            .map(doc => doc.name)
            .join(", ");
            
          finalContent = `I couldn't find any relevant information about "${inputValue}" in the ${
            targetDocumentIds.length === 1 ? 'document' : 'documents'
          } ${docNames}. 
          
          This might be because:
          1. The document doesn't contain information about this specific topic
          2. The query terms don't match the document content closely enough
          3. The document processing might not be complete yet
          
          You could try:
          • Rephrasing your query using different terms
          • Making your query more specific
          • Checking if your document has finished processing
          • Uploading additional documents that cover this topic`;
        }
      }
      
      // Update the loading message with the response
      setMessages(prev => 
        prev.map(msg => 
          msg.id === loadingMessage.id
            ? {
                ...msg,
                content: finalContent,
                loading: false,
                sources: response.chunks?.map(chunk => ({
                  text: chunk.text,
                  score: chunk.score,
                  document_id: chunk.document_id,
                  document_name: chunk.metadata?.document_name || `Document ${chunk.document_id}`,
                })),
                timings: response.timings,
                noResults: !hasResults
              }
            : msg
        )
      );
    } catch (error) {
      console.error('Error querying documents:', error);
      
      // Update the loading message with error
      setMessages(prev => 
        prev.map(msg => 
          msg.id === loadingMessage.id
            ? {
                ...msg,
                content: 'Sorry, an error occurred while processing your query. Please try again.',
                loading: false,
                error: true,
              }
            : msg
        )
      );
      
      toast({
        title: 'Query failed',
        description: 'Failed to get an answer from the system',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  // Handle retry for a failed message
  const handleRetry = async (messageId: string) => {
    // Find the user message that corresponds to this response
    const responseMessage = messages.find(msg => msg.id === messageId);
    if (!responseMessage) return;
    
    // Find the preceding user message
    const userMessageIndex = messages.findIndex(msg => msg.id === messageId) - 1;
    if (userMessageIndex < 0) return;
    
    const userMessage = messages[userMessageIndex];
    
    // Set this message to loading
    setMessages(prev => 
      prev.map(msg => 
        msg.id === messageId
          ? { ...msg, loading: true, error: false, content: '' }
          : msg
      )
    );
    
    setIsLoading(true);
    
    try {
      // Prepare query request
      const queryRequest: QueryRequest = {
        query: userMessage.content,
        rerank,
        top_k: topK,
        show_timings: showTimings,
      };
      
      // Add selected documents if any
      if (selectedDocuments.length > 0) {
        queryRequest.document_ids = selectedDocuments;
      }
      
      // Send query to API
      const response = await ragApi.query(queryRequest);
      
      // Update the message with the response
      setMessages(prev => 
        prev.map(msg => 
          msg.id === messageId
            ? {
                ...msg,
                content: response.response,
                loading: false,
                error: false,
                sources: response.chunks?.map(chunk => ({
                  text: chunk.text,
                  score: chunk.score,
                  document_id: chunk.document_id,
                  document_name: chunk.metadata?.document_name || `Document ${chunk.document_id}`,
                })),
                timings: response.timings,
              }
            : msg
        )
      );
    } catch (error) {
      console.error('Error retrying query:', error);
      
      // Update the message with error
      setMessages(prev => 
        prev.map(msg => 
          msg.id === messageId
            ? {
                ...msg,
                content: 'Sorry, an error occurred while processing your query. Please try again.',
                loading: false,
                error: true,
              }
            : msg
        )
      );
      
      toast({
        title: 'Retry failed',
        description: 'Failed to get an answer from the system',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  // Handle copying message content to clipboard
  const handleCopyMessage = (content: string) => {
    navigator.clipboard.writeText(content);
    toast({
      title: 'Copied to clipboard',
      status: 'success',
      duration: 1500,
      isClosable: true,
    });
  };
  
  // Handle key press in the input field
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };
  
  // Handle document selection change
  const handleDocumentSelectionChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedOptions = Array.from(e.target.selectedOptions, option => option.value);
    setSelectedDocuments(selectedOptions);
  };
  
  return (
    <Box>
      <Flex mb={4} justify="space-between" align="center">
        <Text fontSize="lg" fontWeight="bold">Chat with your documents</Text>
        <Button leftIcon={<FiSettings />} onClick={onSettingsOpen} size="sm">
          Settings
        </Button>
      </Flex>
      
      {/* Settings Drawer */}
      <Drawer isOpen={isSettingsOpen} placement="right" onClose={onSettingsClose}>
        <DrawerOverlay />
        <DrawerContent>
          <DrawerCloseButton />
          <DrawerHeader>Chat Settings</DrawerHeader>
          
          <DrawerBody>
            <VStack spacing={6} align="stretch">
              <FormControl>
                <FormLabel>Rerank Results</FormLabel>
                <Tooltip label="Rerank results for better quality. May be slower.">
                  <Switch
                    isChecked={rerank}
                    onChange={(e) => setRerank(e.target.checked)}
                  />
                </Tooltip>
              </FormControl>
              
              <FormControl>
                <FormLabel>Number of Results (Top K)</FormLabel>
                <Select 
                  value={topK} 
                  onChange={(e) => setTopK(Number(e.target.value))}
                >
                  {[1, 2, 3, 4, 5, 6, 8, 10].map(k => (
                    <option key={k} value={k}>{k}</option>
                  ))}
                </Select>
              </FormControl>
              
              <FormControl>
                <FormLabel>Show Query Timings</FormLabel>
                <Switch
                  isChecked={showTimings}
                  onChange={(e) => setShowTimings(e.target.checked)}
                />
              </FormControl>
              
              <FormControl>
                <FormLabel>Filter by Documents</FormLabel>
                {isLoadingDocuments ? (
                  <Spinner size="sm" />
                ) : (
                  <Select
                    multiple
                    height="120px"
                    value={selectedDocuments}
                    onChange={handleDocumentSelectionChange}
                  >
                    {documents.map(doc => (
                      <option key={doc.id} value={doc.id}>
                        {doc.name}
                      </option>
                    ))}
                  </Select>
                )}
                {selectedDocuments.length > 0 && (
                  <Text fontSize="xs" mt={1}>
                    {selectedDocuments.length} document{selectedDocuments.length > 1 ? 's' : ''} selected
                  </Text>
                )}
              </FormControl>
            </VStack>
          </DrawerBody>
        </DrawerContent>
      </Drawer>
      
      {/* Chat Messages */}
      <Box
        height="60vh"
        overflowY="auto"
        borderWidth={1}
        borderRadius="md"
        p={4}
        mb={4}
        bg="white"
      >
        {messages.length === 0 ? (
          <Flex 
            height="100%" 
            align="center" 
            justify="center" 
            direction="column"
            color="gray.500"
          >
            <Text fontSize="lg" mb={2}>
              No messages yet
            </Text>
            <Text fontSize="sm">
              Start by asking a question about your documents
            </Text>
          </Flex>
        ) : (
          <VStack spacing={4} align="stretch">
            {messages.map((message) => (
              <Box 
                key={message.id}
                alignSelf={message.role === 'user' ? 'flex-end' : 'flex-start'}
                maxWidth="80%"
                width={message.role === 'assistant' ? '80%' : 'auto'}
              >
                <Card 
                  bg={message.role === 'user' ? 'primary.500' : 'white'} 
                  color={message.role === 'user' ? 'white' : 'black'}
                  shadow="md"
                  borderRadius="lg"
                >
                  <CardBody>
                    {message.loading ? (
                      <Flex justify="center" align="center" p={4}>
                        <Spinner mr={3} />
                        <Text>Thinking...</Text>
                      </Flex>
                    ) : (
                      <>
                        <Box mb={message.role === 'assistant' ? 2 : 0}>
                          {message.role === 'assistant' ? (
                            <Box className="markdown-content">
                              <ReactMarkdown>{message.content}</ReactMarkdown>
                            </Box>
                          ) : (
                            <Text>{message.content}</Text>
                          )}
                        </Box>
                        
                        {message.role === 'assistant' && !message.error && (
                          <>
                            {/* Display source information if available */}
                            {message.sources && message.sources.length > 0 && (
                              <Box mt={4}>
                                <Accordion allowToggle>
                                  <AccordionItem border="none">
                                    <AccordionButton 
                                      px={2} 
                                      py={1} 
                                      bg="gray.100" 
                                      borderRadius="md"
                                      _hover={{ bg: 'gray.200' }}
                                    >
                                      <Box flex="1" textAlign="left" fontSize="sm">
                                        Source Documents ({message.sources.length})
                                      </Box>
                                      <AccordionIcon />
                                    </AccordionButton>
                                    <AccordionPanel pb={4}>
                                      <VStack spacing={3} align="stretch">
                                        {message.sources.map((source, index) => (
                                          <Box 
                                            key={index} 
                                            p={2} 
                                            borderWidth={1} 
                                            borderRadius="md"
                                            fontSize="sm"
                                          >
                                            <Flex justify="space-between" mb={1}>
                                              <Badge colorScheme="blue">{source.document_name}</Badge>
                                              <Badge colorScheme="green">
                                                Score: {source.score.toFixed(2)}
                                              </Badge>
                                            </Flex>
                                            <Text fontSize="xs" fontStyle="italic">
                                              {source.text}
                                            </Text>
                                          </Box>
                                        ))}
                                      </VStack>
                                    </AccordionPanel>
                                  </AccordionItem>
                                </Accordion>
                              </Box>
                            )}
                            
                            {/* Display query timings if enabled */}
                            {showTimings && message.timings && (
                              <Box mt={2} fontSize="xs" color="gray.500">
                                <Text>
                                  Retrieved in {(message.timings.retrieval * 1000).toFixed(0)}ms | 
                                  Response in {(message.timings.response * 1000).toFixed(0)}ms | 
                                  Total: {(message.timings.total * 1000).toFixed(0)}ms
                                </Text>
                              </Box>
                            )}
                          </>
                        )}
                      </>
                    )}
                  </CardBody>
                </Card>
                
                {/* Message actions */}
                {message.role === 'assistant' && !message.loading && (
                  <Flex mt={1} justifyContent="flex-end">
                    <IconButton
                      aria-label="Copy message"
                      icon={<FiCopy />}
                      size="xs"
                      variant="ghost"
                      onClick={() => handleCopyMessage(message.content)}
                    />
                    {message.error && (
                      <IconButton
                        aria-label="Retry"
                        icon={<FiRefreshCw />}
                        size="xs"
                        variant="ghost"
                        onClick={() => handleRetry(message.id)}
                        ml={1}
                      />
                    )}
                  </Flex>
                )}
              </Box>
            ))}
            <div ref={messagesEndRef} />
          </VStack>
        )}
      </Box>
      
      {/* Input Area */}
      <Flex>
        <Textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Ask a question about your documents..."
          mr={2}
          resize="none"
          rows={2}
          disabled={isLoading}
        />
        <IconButton
          aria-label="Send message"
          icon={<FiSend />}
          onClick={handleSendMessage}
          isLoading={isLoading}
          disabled={!inputValue.trim()}
          colorScheme="primary"
        />
      </Flex>
    </Box>
  );
};

export default Chat; 