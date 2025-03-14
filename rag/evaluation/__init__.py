"""
Evaluation Module
----------------
Tools for evaluating the performance of the RAG system.

This module provides metrics and evaluation frameworks for assessing
both retrieval quality and generation accuracy.
"""

from rag.evaluation.metrics import (
    calculate_precision,
    calculate_recall,
    calculate_f1,
    calculate_mrr,
    calculate_ndcg,
    calculate_semantic_similarity,
    calculate_factual_consistency
)

from rag.evaluation.evaluator import (
    RAGEvaluator,
    RetrievalTestCase,
    GenerationTestCase,
    EndToEndTestCase,
    EvaluationResult
) 