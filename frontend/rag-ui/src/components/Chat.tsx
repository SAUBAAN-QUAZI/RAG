'use client';

import React, { useState, useRef, useEffect } from 'react';
import { ragApi } from '../api/ragApi';
import ErrorDisplay from './ErrorDisplay';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
}

/**
 * Chat component for querying the RAG system
 */
const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [failedQuery, setFailedQuery] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Send a message to the RAG system
  const sendMessage = async (queryText = input, isRetry = false) => {
    if ((!queryText.trim() && !isRetry) || isLoading) return;

    const query = queryText.trim();
    
    // Don't add a new user message if we're retrying
    if (!isRetry) {
      // Create a new user message
      const userMessage: Message = {
        id: Date.now().toString(),
        content: query,
        sender: 'user',
        timestamp: new Date(),
      };

      // Add the user message to the chat
      setMessages((prev) => [...prev, userMessage]);
      setInput('');
    }
    
    setError(null);
    setIsLoading(true);

    try {
      console.log(`${isRetry ? 'Retrying' : 'Sending'} query: ${query.substring(0, 30)}${query.length > 30 ? '...' : ''}`);
      
      // Query the RAG system
      const response = await ragApi.query({ query });

      // Create a new assistant message
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.answer,
        sender: 'assistant',
        timestamp: new Date(),
      };

      // Add the assistant message to the chat
      setMessages((prev) => [...prev, assistantMessage]);
      
      // Clear any failed query state
      setFailedQuery(null);
      setRetryCount(0);
    } catch (error) {
      console.error('Chat query error:', error);
      
      // Store the failed query for retry
      setFailedQuery(query);
      
      // Increment retry count if this was a retry attempt
      if (isRetry) {
        setRetryCount(prev => prev + 1);
      }
      
      // Set error message with proper formatting
      const errorMessage = error instanceof Error ? error.message : 'Failed to get response';
      setError(errorMessage.replace(/\n/g, '<br>'));
    } finally {
      setIsLoading(false);
    }
  };

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage();
  };
  
  // Handle retry
  const handleRetry = () => {
    if (failedQuery) {
      sendMessage(failedQuery, true);
    }
  };

  // Format timestamp
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="flex flex-col h-[calc(100vh-200px)] max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Chat with RAG</h1>

      {/* Messages container */}
      <div className="flex-grow overflow-y-auto border rounded-md mb-4 p-4 bg-gray-50">
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center text-gray-500">
            <p>Ask a question about your documents</p>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-3 ${
                    message.sender === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-white border text-gray-800'
                  }`}
                >
                  <div className="whitespace-pre-wrap">{message.content}</div>
                  <div
                    className={`text-xs mt-1 ${
                      message.sender === 'user' ? 'text-blue-200' : 'text-gray-500'
                    }`}
                  >
                    {formatTime(message.timestamp)}
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Error message */}
      <ErrorDisplay 
        error={error}
        onRetry={failedQuery ? handleRetry : undefined}
        retryCount={retryCount}
        isRetrying={isLoading && !!failedQuery}
        maxRetries={3}
      />

      {/* Input form */}
      <form onSubmit={handleSubmit} className="flex space-x-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={isLoading}
          className="flex-grow px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          type="submit"
          disabled={!input.trim() || isLoading}
          className={`px-4 py-2 rounded-md text-white font-medium ${
            !input.trim() || isLoading
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {isLoading ? 'Thinking...' : 'Send'}
        </button>
      </form>
    </div>
  );
};

export default Chat; 