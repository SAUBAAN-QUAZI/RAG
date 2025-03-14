# RAG Evaluation Module

This module provides a comprehensive framework for evaluating the performance of Retrieval-Augmented Generation (RAG) systems. It includes metrics for assessing both retrieval and generation quality, as well as tools for analyzing results and identifying areas for improvement.

## Overview

The evaluation module consists of the following components:

- **Metrics**: Functions for calculating various evaluation metrics for retrieval and generation.
- **Test Cases**: Classes for defining test cases for retrieval, generation, and end-to-end evaluation.
- **Evaluator**: A class for running evaluations and aggregating results.
- **Example Scripts**: Scripts demonstrating how to use the evaluation framework.

## Metrics

### Retrieval Metrics

- **Precision**: Proportion of retrieved documents that are relevant.
- **Recall**: Proportion of relevant documents that are retrieved.
- **F1 Score**: Harmonic mean of precision and recall.
- **Mean Reciprocal Rank (MRR)**: Average of the reciprocal ranks of the first relevant document.
- **Normalized Discounted Cumulative Gain (NDCG)**: Measures the ranking quality of the results.

### Generation Metrics

- **Semantic Similarity**: Using embeddings to calculate similarity between two texts.
- **N-gram Overlap**: Measures the overlap between generated text and reference text.
- **Factual Consistency**: Evaluates the consistency of generated text with source context.
- **Answer Relevance**: Measures the relevance of the generated answer to the question.

## Test Cases

### RetrievalTestCase

Used for evaluating retrieval performance. Contains:
- Query
- Relevant chunk IDs
- Optional metadata filters

### GenerationTestCase

Used for evaluating generation performance. Contains:
- Query
- Context
- Expected answer
- Optional description

### EndToEndTestCase

Used for evaluating the entire RAG pipeline. Contains:
- Query
- Relevant chunk IDs
- Expected answer
- Optional metadata filters

## Evaluator

The `RAGEvaluator` class provides methods for:
- Evaluating retrieval performance
- Evaluating generation performance
- Evaluating end-to-end performance
- Saving and loading evaluation results
- Creating and loading test datasets

## Usage Examples

### Basic Retrieval Evaluation

```python
from rag.evaluation import RAGEvaluator, RetrievalTestCase
from rag.retrieval.rag_agent import RAGAgent

# Initialize RAG agent
rag_agent = RAGAgent()

# Create test cases
test_cases = [
    RetrievalTestCase(
        query="What are large language models?",
        relevant_chunk_ids=["chunk1", "chunk2", "chunk3"],
        description="Basic LLM query"
    )
]

# Initialize evaluator
evaluator = RAGEvaluator(rag_agent)

# Evaluate retrieval performance
results = evaluator.evaluate_retrieval(test_cases)

# Print results
print("Retrieval Performance:")
for metric, value in results.metrics.items():
    print(f"  {metric}: {value:.4f}")
```

### Generation Evaluation

```python
from rag.evaluation import RAGEvaluator, GenerationTestCase
from rag.retrieval.rag_agent import RAGAgent

# Initialize RAG agent
rag_agent = RAGAgent()

# Create test cases
test_cases = [
    GenerationTestCase(
        query="What are large language models?",
        context="Large language models (LLMs) are AI systems trained on vast amounts of text data...",
        expected_answer="Large language models are advanced AI systems trained on vast amounts of text...",
        description="LLM definition query"
    )
]

# Initialize evaluator
evaluator = RAGEvaluator(rag_agent)

# Evaluate generation performance
results = evaluator.evaluate_generation(test_cases)

# Print results
print("Generation Performance:")
for metric, value in results.metrics.items():
    print(f"  {metric}: {value:.4f}")
```

### End-to-End Evaluation

```python
from rag.evaluation import RAGEvaluator, EndToEndTestCase
from rag.retrieval.rag_agent import RAGAgent

# Initialize RAG agent
rag_agent = RAGAgent()

# Create test cases
test_cases = [
    EndToEndTestCase(
        query="What are large language models?",
        relevant_chunk_ids=["chunk1", "chunk2", "chunk3"],
        expected_answer="Large language models are advanced AI systems trained on vast amounts of text...",
        description="LLM definition query"
    )
]

# Initialize evaluator
evaluator = RAGEvaluator(rag_agent)

# Evaluate end-to-end performance
results = evaluator.evaluate_end_to_end(test_cases)

# Print results
print("End-to-End Performance:")
for metric, value in results.metrics.items():
    print(f"  {metric}: {value:.4f}")
```

## Example Scripts

The module includes several example scripts:

- `examples/evaluate_llm_book_initial.py`: Basic evaluation of the RAG system using the LLM book PDF.
- `examples/evaluate_llm_book_manual.py`: Evaluation using manual relevance judgments.
- `examples/evaluate_llm_book_comprehensive.py`: Comprehensive evaluation with different chunking strategies and embedding models.
- `examples/evaluate_generation.py`: Evaluation of generation quality.

## Best Practices

1. **Create Diverse Test Cases**: Include a variety of queries covering different topics and complexity levels.
2. **Use Manual Relevance Judgments**: For more accurate evaluation, manually judge the relevance of chunks to queries.
3. **Test Different Configurations**: Experiment with different chunking strategies and embedding models to find the optimal configuration.
4. **Analyze Results**: Use the analysis tools to identify patterns and issues in the system's performance.
5. **Iterate and Improve**: Use the evaluation results to guide improvements to the RAG system.

## Extending the Framework

The evaluation framework can be extended in several ways:

- **Add New Metrics**: Implement additional metrics in the `metrics.py` file.
- **Create Custom Test Cases**: Extend the test case classes to include additional information or functionality.
- **Implement Domain-Specific Evaluations**: Create specialized evaluation scripts for specific domains or use cases.

## Troubleshooting

- **No Chunks Found**: Ensure that documents have been processed and chunks added to the retrieval system before running evaluations.
- **Low Retrieval Performance**: Try different chunking strategies or embedding models to improve retrieval performance.
- **Low Generation Quality**: Check the quality of the retrieved chunks and adjust the generation parameters.
- **Evaluation Errors**: Ensure that the test cases are properly formatted and contain all required information. 