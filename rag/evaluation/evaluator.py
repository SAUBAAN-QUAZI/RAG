"""
RAG Evaluator
-----------
Framework for evaluating the Retrieval-Augmented Generation system.
"""

import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Union, NamedTuple

import numpy as np

from rag.utils import save_json, load_json
from rag.retrieval.rag_agent import RAGAgent
from rag.evaluation.metrics import (
    calculate_precision, calculate_recall, calculate_f1,
    calculate_mrr, calculate_ndcg, calculate_semantic_similarity,
    calculate_factual_consistency, calculate_answer_relevance
)

logger = logging.getLogger("rag.evaluation")

class RetrievalTestCase(NamedTuple):
    """Test case for evaluating retrieval performance."""
    query: str
    relevant_chunk_ids: List[str]
    metadata_filters: Optional[Dict[str, Any]] = None
    description: Optional[str] = None

class GenerationTestCase(NamedTuple):
    """Test case for evaluating generation performance."""
    query: str
    context: str
    expected_answer: str
    description: Optional[str] = None

class EndToEndTestCase(NamedTuple):
    """Test case for evaluating end-to-end RAG performance."""
    query: str
    relevant_chunk_ids: List[str]
    expected_answer: str
    metadata_filters: Optional[Dict[str, Any]] = None
    description: Optional[str] = None

class EvaluationResult(NamedTuple):
    """Results of an evaluation run."""
    metrics: Dict[str, float]
    details: Dict[str, Any]
    timestamp: str = datetime.now().isoformat()

class RAGEvaluator:
    """
    Evaluator for measuring RAG system performance.
    """
    
    def __init__(self, rag_agent: RAGAgent):
        """
        Initialize a RAG evaluator.
        
        Args:
            rag_agent: RAG agent to evaluate
        """
        self.rag_agent = rag_agent
        
    def evaluate_retrieval(self, test_cases: List[RetrievalTestCase], 
                          top_k: int = 10) -> EvaluationResult:
        """
        Evaluate the retrieval component using a set of test cases.
        
        Args:
            test_cases: List of retrieval test cases
            top_k: Number of results to retrieve
            
        Returns:
            EvaluationResult: Evaluation results
        """
        logger.info(f"Evaluating retrieval with {len(test_cases)} test cases")
        
        metrics = {
            "precision": [],
            "recall": [],
            "f1": [],
            "mrr": [],
            "ndcg": [],
            "latency": []
        }
        
        details = {
            "per_query": {}
        }
        
        for tc in test_cases:
            start_time = time.time()
            retriever = self.rag_agent.retriever
            results = retriever.retrieve(tc.query, top_k=top_k, filter_dict=tc.metadata_filters)
            latency = time.time() - start_time
            
            retrieved_ids = {r["chunk_id"] for r in results}
            relevant_ids = set(tc.relevant_chunk_ids)
            
            # Calculate metrics
            precision = calculate_precision(retrieved_ids, relevant_ids)
            recall = calculate_recall(retrieved_ids, relevant_ids)
            f1 = calculate_f1(precision, recall)
            mrr = calculate_mrr(results, relevant_ids)
            ndcg = calculate_ndcg(results, relevant_ids)
            
            # Store metrics
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["f1"].append(f1)
            metrics["mrr"].append(mrr)
            metrics["ndcg"].append(ndcg)
            metrics["latency"].append(latency)
            
            # Store per-query details
            query_key = tc.description or tc.query[:50]
            details["per_query"][query_key] = {
                "query": tc.query,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mrr": mrr,
                "ndcg": ndcg,
                "latency": latency,
                "retrieved_ids": list(retrieved_ids),
                "relevant_ids": list(relevant_ids),
                "results_count": len(results)
            }
        
        # Aggregate metrics
        for key in metrics:
            if metrics[key]:
                metrics[key] = float(np.mean(metrics[key]))
        
        return EvaluationResult(metrics=metrics, details=details)

    def evaluate_generation(self, test_cases: List[GenerationTestCase]) -> EvaluationResult:
        """
        Evaluate the generation component using a set of test cases.
        
        Args:
            test_cases: List of generation test cases
            
        Returns:
            EvaluationResult: Evaluation results
        """
        logger.info(f"Evaluating generation with {len(test_cases)} test cases")
        
        metrics = {
            "factual_consistency": [],
            "answer_relevance": [],
            "semantic_similarity": [],
            "latency": []
        }
        
        details = {
            "per_query": {}
        }
        
        for tc in test_cases:
            start_time = time.time()
            # Use the internal method to generate directly from context
            answer = self.rag_agent._generate_from_context(tc.query, tc.context)
            latency = time.time() - start_time
            
            # Calculate metrics
            factual = calculate_factual_consistency(answer, tc.context)
            relevance = calculate_answer_relevance(answer, tc.query)
            similarity = calculate_semantic_similarity(answer, tc.expected_answer)
            
            # Store metrics
            metrics["factual_consistency"].append(factual)
            metrics["answer_relevance"].append(relevance)
            metrics["semantic_similarity"].append(similarity)
            metrics["latency"].append(latency)
            
            # Store per-query details
            query_key = tc.description or tc.query[:50]
            details["per_query"][query_key] = {
                "query": tc.query,
                "factual_consistency": factual,
                "answer_relevance": relevance,
                "semantic_similarity": similarity,
                "latency": latency,
                "generated_answer": answer,
                "expected_answer": tc.expected_answer,
                "context_length": len(tc.context)
            }
        
        # Aggregate metrics
        for key in metrics:
            if metrics[key]:
                metrics[key] = float(np.mean(metrics[key]))
        
        return EvaluationResult(metrics=metrics, details=details)

    def evaluate_end_to_end(self, test_cases: List[EndToEndTestCase]) -> EvaluationResult:
        """
        Evaluate the entire RAG pipeline using a set of test cases.
        
        Args:
            test_cases: List of end-to-end test cases
            
        Returns:
            EvaluationResult: Evaluation results
        """
        logger.info(f"Evaluating end-to-end with {len(test_cases)} test cases")
        
        metrics = {
            "overall_success": [],
            "factual_consistency": [],
            "answer_relevance": [],
            "semantic_similarity": [],
            "retrieval_precision": [],
            "retrieval_recall": [],
            "latency": []
        }
        
        details = {
            "per_query": {}
        }
        
        for tc in test_cases:
            # Track overall latency
            start_time = time.time()
            
            # Get the rag response
            answer = self.rag_agent.query(tc.query, tc.metadata_filters)
            
            # Separately get retrieval results to evaluate retrieval metrics
            retriever = self.rag_agent.retriever
            retrieval_results = retriever.retrieve(
                tc.query, 
                top_k=self.rag_agent.retriever.top_k,
                filter_dict=tc.metadata_filters
            )
            
            # Get retrieved context to evaluate generation metrics
            context = self.rag_agent.retriever.get_relevant_context(
                tc.query, 
                filter_dict=tc.metadata_filters
            )
            
            latency = time.time() - start_time
            
            # Calculate retrieval metrics
            retrieved_ids = {r["chunk_id"] for r in retrieval_results}
            relevant_ids = set(tc.relevant_chunk_ids)
            precision = calculate_precision(retrieved_ids, relevant_ids)
            recall = calculate_recall(retrieved_ids, relevant_ids)
            
            # Calculate generation metrics
            factual = calculate_factual_consistency(answer, context)
            relevance = calculate_answer_relevance(answer, tc.query)
            similarity = calculate_semantic_similarity(answer, tc.expected_answer)
            
            # Calculate overall success (a combination of metrics)
            # This is a simplified approach - in production you might want a more sophisticated model
            overall_success = 0.4 * similarity + 0.3 * factual + 0.2 * relevance + 0.1 * precision
            
            # Store metrics
            metrics["overall_success"].append(overall_success)
            metrics["factual_consistency"].append(factual)
            metrics["answer_relevance"].append(relevance)
            metrics["semantic_similarity"].append(similarity)
            metrics["retrieval_precision"].append(precision)
            metrics["retrieval_recall"].append(recall)
            metrics["latency"].append(latency)
            
            # Store per-query details
            query_key = tc.description or tc.query[:50]
            details["per_query"][query_key] = {
                "query": tc.query,
                "overall_success": overall_success,
                "factual_consistency": factual,
                "answer_relevance": relevance,
                "semantic_similarity": similarity,
                "retrieval_precision": precision,
                "retrieval_recall": recall,
                "latency": latency,
                "generated_answer": answer,
                "expected_answer": tc.expected_answer,
                "retrieved_ids": list(retrieved_ids),
                "relevant_ids": list(relevant_ids),
                "results_count": len(retrieval_results)
            }
        
        # Aggregate metrics
        for key in metrics:
            if metrics[key]:
                metrics[key] = float(np.mean(metrics[key]))
        
        return EvaluationResult(metrics=metrics, details=details)
    
    def save_results(self, results: EvaluationResult, file_path: Union[str, Path]) -> None:
        """
        Save evaluation results to a file.
        
        Args:
            results: Evaluation results
            file_path: Path to save results
        """
        save_json(results._asdict(), file_path)
        logger.info(f"Saved evaluation results to {file_path}")
    
    def load_results(self, file_path: Union[str, Path]) -> EvaluationResult:
        """
        Load evaluation results from a file.
        
        Args:
            file_path: Path to load results from
            
        Returns:
            EvaluationResult: Loaded evaluation results
        """
        data = load_json(file_path)
        return EvaluationResult(**data)
    
    def create_test_dataset(self, 
                           retrieval_cases: Optional[List[RetrievalTestCase]] = None,
                           generation_cases: Optional[List[GenerationTestCase]] = None,
                           end_to_end_cases: Optional[List[EndToEndTestCase]] = None,
                           file_path: Union[str, Path] = None) -> Dict[str, Any]:
        """
        Create and optionally save a test dataset.
        
        Args:
            retrieval_cases: List of retrieval test cases
            generation_cases: List of generation test cases
            end_to_end_cases: List of end-to-end test cases
            file_path: Optional path to save the dataset
            
        Returns:
            Dict: The created test dataset
        """
        dataset = {
            "retrieval_cases": [tc._asdict() for tc in (retrieval_cases or [])],
            "generation_cases": [tc._asdict() for tc in (generation_cases or [])],
            "end_to_end_cases": [tc._asdict() for tc in (end_to_end_cases or [])],
            "created_at": datetime.now().isoformat()
        }
        
        if file_path:
            save_json(dataset, file_path)
            logger.info(f"Saved test dataset to {file_path}")
            
        return dataset
    
    def load_test_dataset(self, file_path: Union[str, Path]) -> Dict[str, List]:
        """
        Load a test dataset from a file.
        
        Args:
            file_path: Path to the test dataset file
            
        Returns:
            Dict: Dictionary containing test cases
        """
        data = load_json(file_path)
        
        return {
            "retrieval_cases": [RetrievalTestCase(**tc) for tc in data.get("retrieval_cases", [])],
            "generation_cases": [GenerationTestCase(**tc) for tc in data.get("generation_cases", [])],
            "end_to_end_cases": [EndToEndTestCase(**tc) for tc in data.get("end_to_end_cases", [])]
        } 