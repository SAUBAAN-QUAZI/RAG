#!/usr/bin/env python
"""
Example: Comparing RAG System Evaluation Results
------------------------------------------------
This script demonstrates how to compare evaluation results from
multiple runs to track improvements over time.
"""

import os
import sys
import json
import logging
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path if running script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.evaluation import RAGEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def load_results(file_path):
    """Load evaluation results from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading results from {file_path}: {e}")
        return None

def compare_metrics(results_list, names, title, output_path=None):
    """Compare metrics across multiple evaluation runs."""
    if not results_list or not all(results_list):
        logger.error("One or more results could not be loaded")
        return
    
    # Extract metrics from each result
    metrics_data = {}
    for result, name in zip(results_list, names):
        metrics = result.get('metrics', {})
        timestamp = result.get('timestamp', 'Unknown')
        formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M') if isinstance(timestamp, (int, float)) else timestamp
        
        for metric, value in metrics.items():
            if metric not in metrics_data:
                metrics_data[metric] = []
            metrics_data[metric].append((name, value, formatted_time))
    
    # Create comparison plots
    plt.figure(figsize=(12, 8))
    
    # For each metric, create a grouped bar chart
    for i, (metric, data) in enumerate(metrics_data.items()):
        plt.subplot(len(metrics_data), 1, i+1)
        
        run_names = [d[0] for d in data]
        values = [d[1] for d in data]
        
        bars = plt.bar(run_names, values, alpha=0.7)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, value + 0.02,
                    f'{value:.3f}', ha='center', fontsize=9)
        
        plt.title(f"{metric.capitalize()}")
        plt.ylim(0, 1.05)  # Assuming metrics are in range [0, 1]
        
        # Add timestamps as annotations
        for i, (_, _, time_str) in enumerate(data):
            plt.annotate(f"{time_str}", xy=(i, 0.02), 
                       xytext=(0, -25), textcoords='offset points',
                       ha='center', va='top', fontsize=8, rotation=0)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    # Save figure if path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Comparison chart saved to {output_path}")
    
    plt.show()

def create_comparison_table(results_list, names, output_path=None):
    """Create a comparison table of metrics from multiple evaluation runs."""
    if not results_list or not all(results_list):
        logger.error("One or more results could not be loaded")
        return
    
    # Extract metrics from each result
    table_data = {}
    all_metrics = set()
    
    for result, name in zip(results_list, names):
        metrics = result.get('metrics', {})
        timestamp = result.get('timestamp', 'Unknown')
        
        if isinstance(timestamp, (int, float)):
            formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
        else:
            formatted_time = timestamp
            
        table_data[name] = {
            'timestamp': formatted_time,
            **metrics
        }
        all_metrics.update(metrics.keys())
    
    # Create DataFrame
    df = pd.DataFrame.from_dict(table_data, orient='index')
    
    # Ensure all columns are present and sort
    for metric in all_metrics:
        if metric not in df.columns:
            df[metric] = None
    
    # Format numeric columns
    for col in df.columns:
        if col != 'timestamp' and df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    
    # Save to CSV if path provided
    if output_path:
        df.to_csv(output_path)
        logger.info(f"Comparison table saved to {output_path}")
    
    return df

def main():
    """Compare evaluation results from multiple runs."""
    logger.info("Starting evaluation results comparison")
    
    # Define paths to evaluation results
    eval_dir = Path("data/evaluation")
    
    # Example paths - these should be replaced with actual result files
    result_files = [
        eval_dir / "retrieval_results_baseline.json",
        eval_dir / "retrieval_results_improved_embeddings.json", 
        eval_dir / "retrieval_results_with_reranking.json"
    ]
    
    # Check if files exist
    existing_files = [f for f in result_files if f.exists()]
    
    if not existing_files:
        logger.warning(f"No result files found in {eval_dir}")
        logger.info("Creating sample directory structure for future evaluations")
        
        # Create directories
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Please run evaluation tests and save results to compare them")
        return
    
    # Load results
    results = [load_results(f) for f in existing_files]
    names = [f.stem.replace('retrieval_results_', '') for f in existing_files]
    
    # Compare and visualize
    logger.info(f"Comparing results from: {', '.join(names)}")
    
    try:
        # Create output directory for comparison results
        comparison_dir = eval_dir / "comparisons"
        comparison_dir.mkdir(exist_ok=True)
        
        # Generate visualizations
        output_path = comparison_dir / f"retrieval_comparison_{datetime.now().strftime('%Y%m%d')}.png"
        compare_metrics(results, names, "Retrieval Performance Comparison", output_path)
        
        # Generate table
        table_path = comparison_dir / f"retrieval_comparison_{datetime.now().strftime('%Y%m%d')}.csv"
        comparison_table = create_comparison_table(results, names, table_path)
        
        if comparison_table is not None:
            print("\nMetrics Comparison Table:")
            print(comparison_table)
            
    except Exception as e:
        logger.error(f"Error comparing results: {e}")
    
    logger.info("Comparison completed")

if __name__ == "__main__":
    main() 