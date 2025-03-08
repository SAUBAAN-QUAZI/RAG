"""
RAG Agent
-------
This module implements a RAG agent that combines retrieval with LLM generation.
"""

from typing import Dict, List, Optional, Union

from openai import OpenAI

from rag.config import OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP
from rag.document_processing.processor import process_document
from rag.retrieval.retriever import Retriever
from rag.utils import logger, retry_with_exponential_backoff


class RAGAgent:
    """
    RAG (Retrieval-Augmented Generation) Agent.
    
    This agent combines document retrieval with language model generation
    to provide grounded, context-aware responses.
    """
    
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        api_key: str = OPENAI_API_KEY,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500,
    ):
        """
        Initialize a RAG Agent.
        
        Args:
            retriever: Retriever instance for finding relevant documents
            api_key: OpenAI API key
            model: Language model to use for generation
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
        """
        # Create retriever if not provided
        if retriever is None:
            self.retriever = Retriever()
        else:
            self.retriever = retriever
            
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client with compatibility for different environments
        try:
            # Use a simple initialization without any proxy settings
            openai_kwargs = {'api_key': api_key}
            self.client = OpenAI(**openai_kwargs)
            logger.info(f"Successfully initialized OpenAI client for RAG agent")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client in RAG agent: {e}")
            raise
        
        logger.info(f"Initialized RAGAgent with model={model}, "
                   f"temperature={temperature}, max_tokens={max_tokens}")
    
    def add_document(self, file_path: str, **metadata) -> None:
        """
        Add a document to the RAG system.
        
        Args:
            file_path: Path to the document file
            **metadata: Additional metadata to include
        """
        logger.info(f"Adding document to RAG system: {file_path}")
        
        try:
            # Process document with explicit chunk size and overlap to avoid None issues
            # Explicitly pass chunk size and overlap values from config
            result = process_document(
                file_path, 
                chunk_size=CHUNK_SIZE, 
                chunk_overlap=CHUNK_OVERLAP, 
                **metadata
            )
            
            # Log successful processing
            logger.info(f"Successfully processed document: {file_path}")
            logger.info(f"Document ID: {result['document'].doc_id}")
            logger.info(f"Generated {len(result['chunks'])} chunks")
            
            # Add chunks to retriever
            self.retriever.add_chunks(result["chunks"])
            
            logger.info(f"Document {file_path} added successfully to the RAG system")
        except Exception as e:
            logger.exception(f"Error processing document {file_path}: {e}")
            raise
    
    @retry_with_exponential_backoff
    def _generate_from_context(self, query: str, context: str) -> str:
        """
        Generate a response using the language model with context.
        
        Args:
            query: User query
            context: Relevant context for the query
            
        Returns:
            str: Generated response
        """
        # Create system prompt with context
        system_prompt = (
            "You are a helpful AI assistant with access to document information. "
            "Your task is to provide detailed, accurate answers based on the context provided. "
            "The context contains chunks from a document with relevance scores. "
            "If the context contains the information needed to answer the question, use it to provide a comprehensive response. "
            "If asked for a summary, extract the main points and key ideas from all relevant chunks. "
            "If the context doesn't contain enough information, acknowledge the limitations and provide what you can. "
            "Always cite specific documents or page numbers when referencing information."
        )
        
        # Create user message with query and context
        user_message = f"Context:\n{context}\n\nQuestion: {query}"
        
        # Generate response using the language model
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        # Extract and return the generated text
        return response.choices[0].message.content
    
    def query(self, query: str, filter_dict: Optional[Dict] = None) -> str:
        """
        Process a query using the RAG system.
        
        Args:
            query: User query
            filter_dict: Dictionary of metadata filters
            
        Returns:
            str: Generated response
        """
        logger.info(f"Processing query: {query}")
        
        # Check if query is about LLM fundamentals - if so, we can provide a default response
        # even if no context is found (since this is a common query)
        is_llm_fundamentals_query = any(term in query.lower() for term in 
                                       ["llm", "large language model", "fundamental", "basics", "explain"])
        
        # Get relevant context
        context = self.retriever.get_relevant_context(query, filter_dict=filter_dict)
        
        if not context:
            logger.warning("No relevant context found for query")
            
            # Special case for LLM fundamentals
            if is_llm_fundamentals_query:
                logger.info("Query is about LLM fundamentals, providing default response")
                return self._generate_from_context(
                    query,
                    "Based on general knowledge: Large Language Models (LLMs) are AI systems trained on vast amounts "
                    "of text data that can generate human-like text, answer questions, translate languages, write different "
                    "kinds of creative content, and more. They work by predicting the next word in a sequence based on "
                    "the context of previous words. LLMs encode text into numerical vectors (embeddings) and use these "
                    "to generate contextually appropriate responses. They can be enhanced with Retrieval-Augmented "
                    "Generation (RAG) to access specific information not in their training data."
                )
            
            return (
                "I couldn't find any relevant information in the uploaded documents "
                "to answer your question. Could you rephrase or ask something else? "
                "Or, you might want to upload more relevant documents that contain "
                "the information you're looking for."
            )
            
        try:
            # Generate response using the context
            response = self._generate_from_context(query, context)
            return response
        except Exception as e:
            logger.exception(f"Error generating response: {str(e)}")
            return (
                "I encountered an error while generating a response. "
                "This might be due to the complexity of the question or issues with the context. "
                "Please try again with a more specific question or upload different documents."
            ) 