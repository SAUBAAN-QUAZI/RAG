"""
RAG Agent
-------
This module implements a RAG agent that combines retrieval with LLM generation.
"""

from typing import Dict, List, Optional, Union

from openai import OpenAI

from rag.config import OPENAI_API_KEY
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
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
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
        
        # Process document
        result = process_document(file_path, **metadata)
        
        # Add chunks to retriever
        self.retriever.add_chunks(result["chunks"])
    
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
            "You are a helpful assistant that answers questions based on the provided context. "
            "If the context doesn't contain relevant information to answer the question, "
            "say that you don't have enough information to provide a complete answer. "
            "Use the context to ground your answer and cite the specific documents used."
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
        
        # Get relevant context
        context = self.retriever.get_relevant_context(query, filter_dict=filter_dict)
        
        if not context:
            logger.warning("No relevant context found for query")
            return (
                "I couldn't find any relevant information in my knowledge base "
                "to answer your question. Could you rephrase or ask something else?"
            )
            
        # Generate response using the context
        response = self._generate_from_context(query, context)
        
        return response 