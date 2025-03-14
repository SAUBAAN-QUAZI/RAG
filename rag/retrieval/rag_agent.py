"""
RAG Agent Module
-----------
This module implements a RAG (Retrieval-Augmented Generation) agent 
that combines retrieval and generation to answer queries.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import openai
from openai import OpenAI

from rag.config import OPENAI_API_KEY, MAX_RESPONSE_TOKENS
from rag.retrieval.retriever import Retriever
from rag.document_processing.processor import process_document
from rag.utils import logger, retry_with_exponential_backoff

def create_openai_client(api_key=None):
    """Create an OpenAI client with proper error handling."""
    try:
        if not api_key and not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not provided")
            
        return OpenAI(api_key=api_key or OPENAI_API_KEY)
    except Exception as e:
        logger.error(f"Error creating OpenAI client: {e}")
        raise


class RAGAgent:
    """
    RAG (Retrieval-Augmented Generation) Agent.
    
    This agent combines document retrieval with language model generation
    to provide grounded, context-aware responses with enhanced accuracy.
    """
    
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        api_key: str = OPENAI_API_KEY,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = MAX_RESPONSE_TOKENS,
        enable_query_rewriting: bool = True,
        use_structured_prompting: bool = True,
        enable_self_reflection: bool = True,
    ):
        """
        Initialize a RAG Agent with advanced prompting strategies.
        
        Args:
            retriever: Retriever instance for finding relevant documents
            api_key: OpenAI API key
            model: Language model to use for generation
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate (default 1500, configurable via MAX_RESPONSE_TOKENS env var)
            enable_query_rewriting: Whether to rewrite queries for better retrieval
            use_structured_prompting: Whether to use structured prompts for better answers
            enable_self_reflection: Whether to enable self-reflection for better accuracy
        """
        # Create retriever if not provided
        if retriever is None:
            self.retriever = Retriever()
        else:
            self.retriever = retriever
            
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_query_rewriting = enable_query_rewriting
        self.use_structured_prompting = use_structured_prompting
        self.enable_self_reflection = enable_self_reflection
        
        # Initialize OpenAI client with compatibility for different environments
        try:
            # Use our safe initialization helper
            self.client = create_openai_client(api_key)
            logger.info(f"Successfully initialized OpenAI client for RAG agent")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client in RAG agent: {e}")
            raise
        
        logger.info(f"Initialized RAGAgent with model={model}, "
                   f"temperature={temperature}, max_tokens={max_tokens}, "
                   f"query_rewriting={enable_query_rewriting}, "
                   f"structured_prompting={use_structured_prompting}, "
                   f"self_reflection={enable_self_reflection}")
    
    @retry_with_exponential_backoff
    def _rewrite_query(self, query: str) -> str:
        """
        Rewrite the query to make it more effective for retrieval.
        
        Args:
            query: Original user query
            
        Returns:
            str: Rewritten query optimized for retrieval
        """
        if not self.enable_query_rewriting:
            return query
            
        logger.info(f"Rewriting query for better retrieval: {query}")
        
        # Create system prompt for query rewriting
        system_prompt = (
            "You are a query optimization assistant. Your task is to rewrite the user's query "
            "to make it more effective for retrieving relevant information from a vector database. "
            "Your rewritten query should:\n"
            "1. Be more specific and detailed than the original\n"
            "2. Include key entities and concepts related to the query\n"
            "3. Use synonyms for important terms to increase recall\n"
            "4. Expand acronyms and domain-specific terminology\n"
            "5. Be formulated as a clear and detailed information request\n\n"
            "Respond ONLY with the rewritten query, no explanations or other text."
        )
        
        # Create user message with query
        user_message = f"Original query: {query}\n\nPlease rewrite this query to be more effective for retrieval."
        
        try:
            # Generate rewritten query
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,  # Lower temperature for more predictable results
                max_tokens=100,
            )
            
            # Extract and process the rewritten query
            rewritten_query = response.choices[0].message.content.strip()
            
            # Ensure the rewritten query is not too long
            if len(rewritten_query) > 500:
                rewritten_query = rewritten_query[:500] + "..."
                
            logger.info(f"Rewrote query: '{query}' → '{rewritten_query}'")
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}")
            # Fall back to original query if rewriting fails
            return query
    
    @retry_with_exponential_backoff
    def _generate_from_context(
        self, 
        query: str, 
        context: str,
        use_structured_output: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate a response using the language model with enhanced context processing.
        
        Args:
            query: User query
            context: Relevant context for the query
            use_structured_output: Whether to return structured output
            
        Returns:
            str or Dict: Generated response, either as text or structured format
        """
        # Create enhanced system prompt with context
        system_prompt = (
            "You are a highly knowledgeable AI assistant that provides accurate, detailed, and helpful "
            "responses based on the given context. Your expertise lies in extracting relevant information "
            "from documents and synthesizing coherent, comprehensive answers.\n\n"
            "Guidelines for your response:\n"
            "- Base your answer primarily on the provided context, citing specific sources when available\n"
            "- If the context contains contradictory information, acknowledge this and explain the different perspectives\n"
            "- If the context is insufficient to fully answer the question, clearly state what information is missing\n"
            "- Present information in a logical, easy-to-follow structure with headers and bullet points when appropriate\n"
            "- For numerical or statistical information, present it accurately with proper context\n"
            "- If asked for a summary, extract the key points and main ideas from all relevant chunks\n"
            "- Avoid making up information - if you need to supplement with general knowledge, clearly indicate this\n"
            "- If multiple documents are referenced, synthesize the information into a coherent response\n"
            "- Always cite the specific document ID or page number when referencing information\n\n"
            "The context provided contains chunks from documents with relevance scores indicating how closely "
            "they match the query. Higher relevance scores generally indicate more useful information."
        )
        
        # Create user message with query and context
        user_message = f"Context:\n{context}\n\nQuestion: {query}"
        
        # Add structured prompt format if enabled
        if self.use_structured_prompting and use_structured_output:
            system_prompt += (
                "\n\nPlease structure your response in the following JSON format:\n"
                "{\n"
                '  "answer": "The comprehensive answer to the question",\n'
                '  "sources": ["List of document IDs or page numbers used"],\n'
                '  "confidence": "HIGH/MEDIUM/LOW based on the quality of context",\n'
                '  "missing_info": "Any missing information needed for a complete answer"\n'
                "}"
            )
            user_message += "\n\nPlease provide your response in the structured JSON format specified."
        
        # Generate response using the language model
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"} if use_structured_output else None,
            )
            
            # Extract and return the generated text
            generated_text = response.choices[0].message.content
            
            # If structured output is requested, parse as JSON
            if use_structured_output:
                try:
                    return json.loads(generated_text)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse structured output as JSON, returning raw text")
                    return generated_text
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    @retry_with_exponential_backoff
    def _verify_response(self, query: str, context: str, response: str) -> Dict[str, Any]:
        """
        Verify the accuracy and completeness of the generated response.
        
        Args:
            query: Original user query
            context: Context used for generation
            response: Generated response
            
        Returns:
            Dict containing verification results
        """
        if not self.enable_self_reflection:
            return {"verified": True, "confidence": "HIGH", "issues": []}
            
        logger.info("Verifying response accuracy and completeness")
        
        # Create system prompt for verification
        system_prompt = (
            "You are a critical fact-checking assistant. Your task is to assess the accuracy, "
            "completeness, and factual consistency of a generated response against the provided context. "
            "Analyze whether the response accurately reflects information from the context, "
            "whether any information is missing, and whether any statements in the response "
            "are not supported by the context."
        )
        
        # Create user message with query, context, and response
        user_message = (
            f"Original query: {query}\n\n"
            f"Context used for generation:\n{context}\n\n"
            f"Generated response:\n{response}\n\n"
            "Please assess the response and provide your evaluation in the following JSON format:\n"
            "{\n"
            '  "verified": true/false,\n'
            '  "confidence": "HIGH/MEDIUM/LOW",\n'
            '  "issues": ["List any specific issues, inaccuracies, or unsupported claims"],\n'
            '  "missing_info": ["List any important information from the context that was omitted"],\n'
            '  "suggested_improvements": ["List specific improvements that would make the response better"]\n'
            "}"
        )
        
        try:
            # Generate verification
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,  # Lower temperature for more reliable verification
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            
            # Extract and parse the verification result
            verification_text = response.choices[0].message.content
            try:
                verification_result = json.loads(verification_text)
                logger.info(f"Response verification result: confidence={verification_result.get('confidence', 'UNKNOWN')}, verified={verification_result.get('verified', False)}")
                return verification_result
            except json.JSONDecodeError:
                logger.warning("Failed to parse verification result as JSON")
                return {"verified": True, "confidence": "UNKNOWN", "issues": ["Verification failed"]}
                
        except Exception as e:
            logger.error(f"Error verifying response: {str(e)}")
            return {"verified": True, "confidence": "UNKNOWN", "issues": [f"Verification error: {str(e)}"]}
    
    @retry_with_exponential_backoff
    def _improve_response(self, query: str, context: str, original_response: str, verification: Dict[str, Any]) -> str:
        """
        Improve the response based on verification feedback.
        
        Args:
            query: Original user query
            context: Context used for generation
            original_response: Generated response
            verification: Verification results
            
        Returns:
            str: Improved response
        """
        if not verification.get("issues") and not verification.get("missing_info"):
            return original_response
            
        logger.info("Improving response based on verification feedback")
        
        # Extract issues and improvements
        issues = verification.get("issues", [])
        missing_info = verification.get("missing_info", [])
        suggested_improvements = verification.get("suggested_improvements", [])
        
        # Create system prompt for improvement
        system_prompt = (
            "You are a helpful assistant that improves responses based on feedback. "
            "Your task is to revise the original response to address identified issues, "
            "incorporate missing information, and implement suggested improvements. "
            "Ensure the revised response is accurate, complete, and well-structured."
        )
        
        # Create user message with query, context, response, and feedback
        user_message = (
            f"Original query: {query}\n\n"
            f"Context:\n{context}\n\n"
            f"Original response:\n{original_response}\n\n"
            "Feedback for improvement:\n"
        )
        
        if issues:
            user_message += "Issues to fix:\n" + "\n".join([f"- {issue}" for issue in issues]) + "\n\n"
        
        if missing_info:
            user_message += "Missing information to add:\n" + "\n".join([f"- {info}" for info in missing_info]) + "\n\n"
        
        if suggested_improvements:
            user_message += "Suggested improvements:\n" + "\n".join([f"- {improvement}" for improvement in suggested_improvements]) + "\n\n"
        
        user_message += "Please provide an improved response that addresses all the feedback."
        
        try:
            # Generate improved response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract and return the improved response
            improved_response = response.choices[0].message.content
            logger.info("Successfully improved response based on verification feedback")
            return improved_response
            
        except Exception as e:
            logger.error(f"Error improving response: {str(e)}")
            # Fall back to original response if improvement fails
            return original_response
    
    def query(self, query: str, filter_dict: Optional[Dict] = None) -> str:
        """
        Process a query using the RAG system with enhanced prompting and verification.
        
        Args:
            query: User query
            filter_dict: Dictionary of metadata filters
            
        Returns:
            str: Generated response
        """
        logger.info(f"Processing query: {query}")
        start_time = datetime.now()
        
        # Step 1: Rewrite query for better retrieval if enabled
        retrieval_query = self._rewrite_query(query) if self.enable_query_rewriting else query
        
        # Step 2: Check if query is about LLM fundamentals for fallback response
        is_llm_fundamentals_query = any(term in query.lower() for term in 
                                      ["llm", "large language model", "fundamental", "basics", "explain"])
        
        # Step 3: Get relevant context using retrieval query
        context = self.retriever.get_relevant_context(retrieval_query, filter_dict=filter_dict)
        
        # Step 4: Handle case where no context is found
        if not context:
            logger.warning("No relevant context found for query")
            
            # Special case for LLM fundamentals
            if is_llm_fundamentals_query:
                logger.info("Query is about LLM fundamentals, providing default response")
                fallback_context = (
                    "Based on general knowledge: Large Language Models (LLMs) are AI systems trained on vast amounts "
                    "of text data that can generate human-like text, answer questions, translate languages, write different "
                    "kinds of creative content, and more. They work by predicting the next word in a sequence based on "
                    "the context of previous words. LLMs encode text into numerical vectors (embeddings) and use these "
                    "to generate contextually appropriate responses. They can be enhanced with Retrieval-Augmented "
                    "Generation (RAG) to access specific information not in their training data."
                )
                return self._generate_from_context(query, fallback_context)
            
            # Generic no-context response
            return (
                "I couldn't find any relevant information in the available documents "
                "to answer your question. Could you rephrase your question or provide more details? "
                "Alternatively, more relevant documents may need to be added to the knowledge base."
            )
        
        try:
            # Step 5: Generate initial response with structured format if enabled
            use_structured = self.use_structured_prompting and self.enable_self_reflection
            initial_response = self._generate_from_context(query, context, use_structured_output=use_structured)
            
            # Extract the text response if structured format was used
            if isinstance(initial_response, dict):
                response_text = initial_response.get("answer", str(initial_response))
                # Log additional structured information
                logger.info(f"Structured response received with confidence: {initial_response.get('confidence', 'N/A')}")
                if 'sources' in initial_response:
                    logger.info(f"Sources used: {initial_response.get('sources', [])}")
            else:
                response_text = initial_response
            
            # Step 6: Verify and improve response if self-reflection is enabled
            if self.enable_self_reflection:
                # Verify response accuracy and completeness
                verification = self._verify_response(query, context, response_text)
                
                # Improve response if verification found issues
                if not verification.get("verified", True) or verification.get("confidence", "HIGH") != "HIGH":
                    logger.info(f"Response verification found issues, improving response")
                    response_text = self._improve_response(query, context, response_text, verification)
            
            # Log query processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Query processed in {processing_time:.2f} seconds")
            
            # Always return a string response
            return response_text
            
        except Exception as e:
            logger.exception(f"Error generating response: {str(e)}")
            return (
                "I encountered an error while generating a response. "
                "This might be due to the complexity of the question or issues with the context. "
                "Please try again with a more specific question or contact support if the issue persists."
            )
    
    def add_document(self, file_path: Union[str, Path], **metadata) -> Dict[str, Any]:
        """
        Process and add a document to the RAG system.
        
        Args:
            file_path: Path to the document file
            **metadata: Additional metadata for the document (title, author, etc.)
            
        Returns:
            Dict containing document info and processing status
            
        Raises:
            Exception: If document processing fails
        """
        logger.info(f"Adding document: {file_path}")
        
        # Process the document to extract chunks
        try:
            # Use the document processor to split document into chunks
            processing_result = process_document(
                file_path=file_path,
                save_results=True,
                **metadata
            )
            
            document = processing_result["document"]
            chunks = processing_result["chunks"]
            
            logger.info(f"Document processed: {document.doc_id}, generated {len(chunks)} chunks")
            
            # Add chunks to the retrieval system
            if self.retriever:
                self.retriever.add_chunks(chunks)
                logger.info(f"Added {len(chunks)} chunks to retrieval system")
            else:
                logger.warning("Retriever not available, chunks not added to retrieval system")
            
            return {
                "status": "success",
                "document_id": document.doc_id,
                "chunk_count": len(chunks),
                "metadata": document.metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            logger.exception("Document processing exception:")
            raise  # Re-raise the exception for the caller to handle 