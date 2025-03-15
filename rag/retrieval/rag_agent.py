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
import time
import re

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
        Rewrite a query to improve retrieval performance.
        
        This method intelligently determines when to rewrite a query and when to leave it as is,
        based on query characteristics. It preserves simple, direct queries and only enhances
        complex or ambiguous ones.
        
        Args:
            query: Original user query
            
        Returns:
            str: Rewritten query or original if rewriting not needed
        """
        # Skip rewriting for very short or simple queries (likely already specific)
        if len(query.split()) <= 5 or query.endswith('?'):
            logger.info(f"Query '{query}' is simple and specific - skipping rewrite")
            return query
            
        # Skip rewriting for queries that appear to target specific sections
        section_indicators = ["chapter", "section", "page", "paragraph", "appendix"]
        if any(indicator in query.lower() for indicator in section_indicators):
            logger.info(f"Query '{query}' appears to target a specific document section - skipping rewrite")
            return query
            
        # Proceed with rewriting for more complex or ambiguous queries
        try:
            # Create a more domain-aware rewriting prompt
            rewrite_prompt = f"""
            You are helping improve search queries for a technical document retrieval system.
            Original query: "{query}"
            
            Guidelines for rewriting:
            1. Preserve the technical intent and domain specificity of the original query
            2. Expand abbreviations and add key technical terms that might be in the documents
            3. DO NOT transform the query into a generic literary or essay analysis prompt
            4. DO NOT add fictional elements (like "character development" for technical topics)
            5. Keep the rewrite focused and concise - don't create an overly verbose query
            
            Rewritten query:
            """
            
            # Use chat completion for query rewriting
            messages = [
                {"role": "system", "content": "You are a technical search query optimization expert. Your task is to rewrite search queries to improve document retrieval without changing their core technical intent."},
                {"role": "user", "content": rewrite_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,  # Use the same model as the main system
                messages=messages,
                temperature=0.0,  # Low temperature for consistent rewrites
                max_tokens=150  # Keep rewrites relatively short
            )
            
            rewritten_query = response.choices[0].message.content.strip()
            
            # Verify the rewritten query isn't wildly different
            # Skip rewrite if it's much longer than the original (sign of over-expansion)
            original_token_count = len(query.split())
            rewritten_token_count = len(rewritten_query.split())
            
            # Reject rewrite if it grows too much
            if rewritten_token_count > 2 * original_token_count and rewritten_token_count > 12:
                logger.warning(f"Rewritten query rejected - too verbose ({original_token_count} → {rewritten_token_count} tokens)")
                return query
                
            # Return the rewritten query
            logger.info(f"Rewrote query: '{query}' → '{rewritten_query}'")
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            # Return the original query if rewriting fails
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
    
    @retry_with_exponential_backoff
    def _assess_context_relevance(self, query: str, context: str) -> Dict[str, Any]:
        """
        Assess the relevance of retrieved context to the query.
        
        This evaluates whether the retrieved context is actually useful for answering
        the user's query, preventing unhelpful responses when context is irrelevant.
        
        Args:
            query: The user's query
            context: The retrieved context
            
        Returns:
            Dict containing relevance assessment with scores and explanation
        """
        if not context or len(context.strip()) < 50:
            return {
                "is_relevant": False,
                "relevance_score": 0.0,
                "explanation": "Retrieved context is empty or too short."
            }
            
        # Create a prompt for assessing relevance
        assessment_prompt = f"""
        You are an expert at evaluating search result relevance. Assess how relevant the retrieved context is for answering the user's query.
        
        User Query: {query}
        
        Retrieved Context: 
        {context[:2000]}... [context truncated for brevity]
        
        First, determine if the context contains information that would help answer the query.
        Then, assign a relevance score from 0 to 10, where:
        - 0-3: Context is not relevant to the query
        - 4-6: Context is somewhat relevant but missing key information
        - 7-10: Context is highly relevant and sufficient to answer the query
        
        Output your assessment in JSON format:
        {{
            "is_relevant": true/false,
            "relevance_score": number,
            "explanation": "Your explanation of the assessment"
        }}
        """
        
        try:
            # Get assessment from LLM
            messages = [
                {"role": "system", "content": "You are an expert at evaluating document relevance. Provide honest assessments in the exact JSON format requested."},
                {"role": "user", "content": assessment_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,  # Keep temperature low for consistent evaluations
                response_format={"type": "json_object"},
                max_tokens=300
            )
            
            # Extract and parse the response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Ensure required fields are present
            required_fields = ["is_relevant", "relevance_score", "explanation"]
            for field in required_fields:
                if field not in result:
                    result[field] = False if field == "is_relevant" else (0.0 if field == "relevance_score" else "Assessment incomplete")
                    
            # Log the assessment
            logger.info(f"Context relevance assessment: score={result['relevance_score']}, is_relevant={result['is_relevant']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error assessing context relevance: {e}")
            # Return default values if assessment fails
            return {
                "is_relevant": True,  # Default to assuming relevance to not block responses
                "relevance_score": 5.0,
                "explanation": f"Failed to assess relevance due to error: {str(e)}"
            }
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess and optimize the query before retrieval.
        
        This method analyzes the query and applies various optimizations:
        1. For chapter/section queries - detects and formats appropriately
        2. For factual queries - keeps concise and focused
        3. For complex queries - ensures clarity and specificity
        
        Args:
            query: The original user query
            
        Returns:
            Processed query optimized for retrieval
        """
        # Clean up the query
        clean_query = query.strip()
        
        # Check for chapter/section references
        chapter_match = re.search(r"chapter\s+(\d+|[ivxlcdm]+)", clean_query, re.IGNORECASE)
        section_match = re.search(r"section\s+(\d+(\.\d+)*)", clean_query, re.IGNORECASE)
        
        if chapter_match or section_match:
            # This is a query about a specific document section
            logger.info(f"Query '{clean_query}' appears to target a specific document section - skipping rewrite")
            return clean_query
            
        # Check if rewrite is needed
        if len(clean_query.split()) <= 2 or clean_query.endswith('?'):
            # Short queries or questions don't need rewriting
            return clean_query
            
        # Use a basic rewrite for complex queries
        if len(clean_query.split()) > 10 and not any(x in clean_query.lower() for x in ['?', 'what', 'how', 'why', 'when', 'where', 'who']):
            # Long non-question query - convert to question form
            if clean_query.lower().startswith('tell me about'):
                # Already in a good form
                return clean_query
            else:
                # Make it a request for information
                return f"Tell me about {clean_query}"
                
        # Default - return original query
        return clean_query

    def query(self, query: str, filters: Optional[Dict] = None) -> Union[str, Dict[str, Any]]:
        """
        Process a query using the RAG system.
        
        Args:
            query: The user's query
            filters: Optional metadata filters for document retrieval
            
        Returns:
            Response text or a dictionary with answer and additional information
        """
        try:
            # Track start time for performance measurement
            self.start_time = time.time()
            
            logger.info(f"Processing query: {query}")
            
            # Preprocess the query
            processed_query = self._preprocess_query(query)
            
            # Consider query rewriting if enabled and not a structural query
            if self.enable_query_rewriting:
                # Check if the query refers to a specific document section (skip rewrite in that case)
                if "chapter" not in processed_query.lower() and "section" not in processed_query.lower():
                    rewritten_query = self._rewrite_query(processed_query)
                    if rewritten_query != processed_query:
                        logger.info(f"Rewrote query: '{processed_query}' → '{rewritten_query}'")
                        processed_query = rewritten_query
                else:
                    logger.info(f"Query '{processed_query}' appears to target a specific document section - skipping rewrite")
            else:
                # Skip rewriting - just log
                logger.info(f"Rewrote query: '{processed_query}' → '{processed_query}'")
            
            # Get relevant context
            logger.info(f"Retrieving documents for query: {processed_query}")
            
            # Retrieve relevant context
            retrieval_start_time = time.time()
            context = self.retriever.get_relevant_context(
                query=processed_query,
                filter_dict=filters
            )
            retrieval_time = time.time() - retrieval_start_time
            logger.info(f"Retrieved context in {retrieval_time:.2f} seconds")
            
            if not context.strip():
                response = {
                    "answer": "I don't have enough relevant information to answer this query.",
                    "sources": [],
                    "confidence": "LOW"
                }
                return response
            
            # Optional context relevance check
            is_context_relevant = True
            if hasattr(self, 'assess_relevance') and self.assess_relevance and hasattr(self, 'context_assessor'):
                assessment_start_time = time.time()
                assessment = self.context_assessor.assess_relevance(processed_query, context)
                is_context_relevant = assessment["is_relevant"]
                logger.info(f"Context relevance assessment: score={assessment['score']}, is_relevant={is_context_relevant}")
                logger.info(f"Context assessment completed in {time.time() - assessment_start_time:.2f} seconds")
                
                if not is_context_relevant:
                    logger.warning(f"Retrieved context is not relevant to the query. Score: {assessment['score']}")
                    # Return a response indicating lack of relevant information
                    response = {
                        "answer": "The provided context does not contain information relevant to your query about '{}'".format(query),
                        "sources": [],
                        "confidence": "LOW"
                    }
                    return response
            
            # Complete response with context using augmented generation
            generation_start_time = time.time()
            answer = self._generate_from_context(
                query=processed_query,
                context=context,
                use_structured_output=self.use_structured_prompting
            )
            generation_time = time.time() - generation_start_time
            logger.info(f"Generated response in {generation_time:.2f} seconds")
            
            # Construct result
            if isinstance(answer, dict):
                # LLM service already returned a structured response
                response = answer
                if "sources" not in response:
                    response["sources"] = []
            else:
                # Simple text response, wrap in a dict
                response = {
                    "answer": answer,
                    "sources": [],  # Would need additional processing to extract
                    "confidence": "MEDIUM"
                }
            
            # Log total query processing time
            total_time = time.time() - self.start_time
            logger.info(f"Total query processing completed in {total_time:.2f} seconds")
            
            return response
            
        except Exception as e:
            # Calculate time until error occurred
            error_time = time.time() - self.start_time if hasattr(self, 'start_time') else 0
            logger.exception(f"Error in RAG query after {error_time:.2f} seconds: {e}")
            # Return a friendly error message
            return {
                "answer": f"I encountered an error while processing your query. Please try again or rephrase your question.",
                "error": str(e),
                "confidence": "LOW"
            }
    
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