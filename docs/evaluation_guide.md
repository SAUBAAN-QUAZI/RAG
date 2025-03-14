# RAG System Evaluation Guide

This guide outlines a comprehensive process for evaluating the performance of your Retrieval-Augmented Generation (RAG) system using the evaluation framework provided in the `rag/evaluation` module.

## Table of Contents

1. [Overview](#overview)
2. [Setting Up](#setting-up)
3. [Creating Test Datasets](#creating-test-datasets)
4. [Evaluating Retrieval Performance](#evaluating-retrieval-performance)
5. [Evaluating Generation Performance](#evaluating-generation-performance)
6. [End-to-End Evaluation](#end-to-end-evaluation)
7. [Analyzing Results](#analyzing-results)
8. [Tracking Performance Over Time](#tracking-performance-over-time)
9. [Improving Your RAG System](#improving-your-rag-system)
10. [Advanced Evaluation Techniques](#advanced-evaluation-techniques)

## Overview

Effective evaluation of a RAG system involves assessing both components:

1. **Retrieval Component**: How well does the system retrieve relevant documents?
2. **Generation Component**: How well does the system generate accurate, relevant responses using the retrieved context?

The evaluation framework provides metrics for both components and allows end-to-end assessment of the entire pipeline.

## Setting Up

First, make sure the evaluation module is installed and properly configured:

```python
from rag.retrieval.rag_agent import RAGAgent
from rag.evaluation import RAGEvaluator

# Initialize your RAG agent with your configuration
rag_agent = RAGAgent()

# Create an evaluator
evaluator = RAGEvaluator(rag_agent)
```

## Creating Test Datasets

A good test dataset is crucial for effective evaluation. It should cover a wide range of query types and expected results.

### Types of Test Cases

1. **Retrieval Test Cases**: Queries with known relevant document IDs
2. **Generation Test Cases**: Queries with context and expected answers
3. **End-to-End Test Cases**: Queries with both relevant document IDs and expected answers

### Creating a Balanced Test Dataset

Your test dataset should include:

- **Factoid Queries**: Questions with definitive answers
- **Exploratory Queries**: Open-ended questions requiring longer explanations
- **Edge Cases**: Rare or difficult queries that test the system's limits
- **Domain-Specific Queries**: Questions relevant to your specific domain

### Example: Creating a Test Dataset

```python
from rag.evaluation import RetrievalTestCase, GenerationTestCase, EndToEndTestCase
from pathlib import Path

# Create test cases
retrieval_cases = [
    RetrievalTestCase(
        query="What is a transformer architecture?",
        relevant_chunk_ids=["doc1_chunk3", "doc2_chunk5"],
        description="Basic transformer definition query"
    ),
    # Add more test cases...
]

# Save the dataset
dataset_path = Path("data/evaluation/test_dataset.json")
dataset_path.parent.mkdir(parents=True, exist_ok=True)

evaluator.create_test_dataset(
    retrieval_cases=retrieval_cases,
    generation_cases=[],  # Add generation test cases if needed
    end_to_end_cases=[],  # Add end-to-end test cases if needed
    file_path=dataset_path
)
```

## Evaluating Retrieval Performance

Retrieval evaluation measures how well your system retrieves relevant documents for a given query.

### Key Metrics

- **Precision**: Proportion of retrieved documents that are relevant
- **Recall**: Proportion of relevant documents that are retrieved
- **F1 Score**: Harmonic mean of precision and recall
- **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks of first relevant results
- **Normalized Discounted Cumulative Gain (NDCG)**: Measures ranking quality

### Running Retrieval Evaluation

```python
# Load dataset or create test cases
test_cases = [
    # ... your test cases
]

# Run evaluation
results = evaluator.evaluate_retrieval(test_cases)

# Print results
print("Retrieval Performance:")
for metric, value in results.metrics.items():
    print(f"  {metric}: {value:.4f}")

# Save results
from pathlib import Path
results_path = Path("data/evaluation/retrieval_results.json")
evaluator.save_results(results, results_path)
```

## Evaluating Generation Performance

Generation evaluation assesses how well your system generates answers based on given context.

### Key Metrics

- **Semantic Similarity**: How similar is the generated answer to the expected answer?
- **Factual Consistency**: How factually accurate is the generated answer?
- **Answer Relevance**: How relevant is the answer to the question?

### Running Generation Evaluation

```python
# Create test cases
test_cases = [
    GenerationTestCase(
        query="Explain embeddings",
        context="Embeddings are dense vector representations...",
        expected_answer="Embeddings are vector representations that capture meaning...",
        description="Embeddings explanation"
    ),
    # Add more test cases...
]

# Run evaluation
results = evaluator.evaluate_generation(test_cases)

# Print results
print("Generation Performance:")
for metric, value in results.metrics.items():
    print(f"  {metric}: {value:.4f}")

# Save results
results_path = Path("data/evaluation/generation_results.json")
evaluator.save_results(results, results_path)
```

## End-to-End Evaluation

End-to-end evaluation assesses the complete RAG pipeline, from query to final answer.

### Running End-to-End Evaluation

```python
# Create test cases
test_cases = [
    EndToEndTestCase(
        query="How does RAG work?",
        relevant_chunk_ids=["chunk345", "chunk678"],
        expected_answer="RAG combines retrieval with generation...",
        description="RAG explanation"
    ),
    # Add more test cases...
]

# Run evaluation
results = evaluator.evaluate_end_to_end(test_cases)

# Print results
print("End-to-End Performance:")
for metric, value in results.metrics.items():
    print(f"  {metric}: {value:.4f}")
```

## Analyzing Results

Beyond aggregate metrics, analyze performance on a per-query basis to identify specific areas for improvement.

### Per-Query Analysis

```python
# Analyze detailed results
details = results.details

for detail in details:
    query = detail.get('query')
    metrics = detail.get('metrics', {})
    
    print(f"\nQuery: {query}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Identify issues
    if metrics.get('precision', 0) < 0.5:
        print("  → Poor precision: Retrieval returning irrelevant documents")
    if metrics.get('recall', 0) < 0.5:
        print("  → Poor recall: Missing relevant documents")
    if metrics.get('factual_consistency', 0) < 0.7:
        print("  → Factual issues: Answer contains inaccuracies")
```

## Tracking Performance Over Time

Track performance over time to identify trends and validate improvements:

```python
# Use examples/compare_evaluation_results.py
# Compare baseline with improvements
```

### Versioning Test Results

It's valuable to save evaluation results with version information:

```python
# Save with version info
version = "v1.2.3"  # Your system version
results_path = Path(f"data/evaluation/retrieval_results_{version}.json")
evaluator.save_results(results, results_path)
```

## Improving Your RAG System

Based on evaluation results, consider improvements:

### Retrieval Improvements

- **Low Precision**: 
  - Implement re-ranking of retrieved documents
  - Adjust similarity thresholds
  - Experiment with different embedding models

- **Low Recall**:
  - Review chunking strategy (chunk size, overlap)
  - Consider hybrid retrieval approaches
  - Try different embedding models with better semantic understanding

### Generation Improvements

- **Low Factual Consistency**:
  - Improve prompt engineering
  - Add fact verification steps
  - Include more context in the generation prompt

- **Low Answer Relevance**:
  - Refine context selection
  - Improve prompt structure
  - Implement query-focused summarization

## Advanced Evaluation Techniques

Consider these advanced techniques for more sophisticated evaluation:

### Human Evaluation

Incorporate human feedback for qualitative assessment:

1. Create a simple web interface for evaluators
2. Ask them to rate answers on dimensions like:
   - Relevance (1-5)
   - Factual accuracy (1-5)
   - Completeness (1-5)
   - Clarity (1-5)

### A/B Testing

Compare different configurations:

1. Split user queries between different RAG configurations
2. Collect metrics on user satisfaction and system performance
3. Use statistical methods to determine significant differences

### Continuous Evaluation

Set up continuous evaluation:

1. Maintain a growing test dataset
2. Run automated evaluation on each system change
3. Track performance trends over time
4. Set up alerts for performance regressions

## Best Practices

1. **Realistic Test Data**: Use real-world queries that represent actual use cases
2. **Comprehensive Test Set**: Cover various query types and difficulty levels
3. **Regular Evaluation**: Evaluate performance after significant changes
4. **Balanced Metrics**: Don't optimize for a single metric at the expense of others
5. **User-Focused**: Remember that technical metrics are proxies for user satisfaction

By following this guide, you'll establish a robust evaluation process that helps you continuously improve your RAG system's performance and reliability. 