import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import ast
import nltk
from nltk.tokenize import word_tokenize
import argparse
from pathlib import Path
from datetime import datetime

def string_to_list(text: str) -> List[str]:
    """Convert string representation of list to actual list"""
    if isinstance(text, str):
        try:
            if text.startswith('[') and text.endswith(']'):
                return ast.literal_eval(text)
            return [text]
        except:
            return [text]
    return text if isinstance(text, list) else [str(text)]

def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    
    text = str(text).lower()
    # Split on whitespace and remove punctuation
    words = text.split()
    return [word.strip('.,;:!?()[]{}""\'') for word in words]

def calculate_metrics(ground_truth: str, prediction: str, is_numeric: bool = False) -> Dict[str, float]:
    """Calculate accuracy, precision, recall, and F1 score"""
    if is_numeric:
        # For numeric fields, use exact matching with tolerance for floating point
        try:
            gt_val = float(ground_truth)
            pred_val = float(prediction)
            # Consider equal if both are 999 (not mentioned) or if they're within 0.01
            is_equal = (gt_val == 999 and pred_val == 999) or abs(gt_val - pred_val) < 0.01
            return {
                'accuracy': float(is_equal),
                'precision': float(is_equal),
                'recall': float(is_equal),
                'f1': float(is_equal)
            }
        except (ValueError, TypeError):
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Convert to lists and tokenize all fields (both text and list fields)
    gt_items = string_to_list(ground_truth)
    pred_items = string_to_list(prediction)
    
    # For both text fields and list fields, use token-level matching
    gt_tokens = set()
    pred_tokens = set()
    
    # Tokenize each item in the lists
    for item in gt_items:
        gt_tokens.update(tokenize_text(item))
    for item in pred_items:
        pred_tokens.update(tokenize_text(item))
    
    # Handle empty cases
    if not gt_tokens and not pred_tokens:
        return {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    if not gt_tokens or not pred_tokens:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    # Calculate true positives, false positives, false negatives
    true_positives = len(gt_tokens.intersection(pred_tokens))
    false_positives = len(pred_tokens - gt_tokens)
    false_negatives = len(gt_tokens - pred_tokens)
    
    # Calculate metrics
    accuracy = true_positives / (true_positives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_csvs(ground_truth_path: str, test_path: str) -> Dict[str, Dict[str, float]]:
    """Evaluate predictions against ground truth"""

    gt_df = pd.read_csv(ground_truth_path)
    test_df = pd.read_csv(test_path)

    # Define fields and their types (text, list, or numeric)
    fields = {
        # Text and list fields (all use token-level matching)
        'primary_disease': False,
        'conditioning_regimen_type': False,
        'conditioning_regimen': False,
        'donor_type': False,
        'transplant_related_complications': False,
        'reason_for_admission': False,
        'problem_list': False,
        'medications_admission': False,
        'medications_discharge': False,
        # Numeric fields (use exact matching)
        'wbc_admission': True,
        'wbc_discharge': True,
        'neuts_admission': True,
        'neuts_discharge': True,
        'hgb_admission': True,
        'hgb_discharge': True,
        'plt_admission': True,
        'plt_discharge': True,
        't_bili_admission': True,
        't_bili_discharge': True,
        'ca_admission': True,
        'ca_discharge': True
    }

    results = {}
    all_metrics = []

    # Calculate metrics for each field
    for field, is_numeric in fields.items():
        if field in gt_df.columns and field in test_df.columns:
            field_metrics = []
            for gt, pred in zip(gt_df[field], test_df[field]):
                metrics = calculate_metrics(gt, pred, is_numeric)
                field_metrics.append(metrics)
                all_metrics.append(metrics)
            
            # Average metrics for this field
            results[field] = {
                metric: np.mean([m[metric] for m in field_metrics])
                for metric in ['accuracy', 'precision', 'recall', 'f1']
            }

    # Calculate overall metrics
    results['overall'] = {
        metric: np.mean([m[metric] for m in all_metrics])
        for metric in ['accuracy', 'precision', 'recall', 'f1']
    }

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate extraction results against ground truth')
    parser.add_argument('--ground-truth', type=str, required=True,
                      help='Path to ground truth CSV file')
    parser.add_argument('--test', type=str, required=True,
                      help='Path to test CSV file')
    
    args = parser.parse_args()
    
    # Validate file paths
    ground_truth_path = Path(args.ground_truth)
    test_path = Path(args.test)
    
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    results = evaluate_csvs(str(ground_truth_path), str(test_path))
    
    # Save results to CSV
    output_path = Path('results')
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f'evaluation_results_{timestamp}.csv'
    
    # Convert results to DataFrame format
    results_data = []
    for field, metrics in results.items():
        row = {'field': field}
        row.update(metrics)
        results_data.append(row)
    
    pd.DataFrame(results_data).to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    print("\nEvaluation Metrics by Field:")
    print("-" * 50)
    
    for field, metrics in results.items():
        if field != 'overall':
            print(f"\n{field}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    print("\nOVERALL METRICS:")
    print("-" * 50)
    for metric, value in results['overall'].items():
        print(f"{metric}: {value:.4f}")
