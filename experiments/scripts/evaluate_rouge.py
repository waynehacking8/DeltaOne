#!/usr/bin/env python3
"""
ROUGE Evaluation for Utility (Summarization) Task
Evaluates the quality of model-generated summaries against ground truth
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd

def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def compute_rouge_basic(prediction: str, reference: str) -> Dict[str, float]:
    """
    Compute basic ROUGE scores using simple token overlap
    (Simple implementation - for production use rouge_score library)
    """
    def tokenize(text):
        return text.lower().split()

    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    # ROUGE-1: Unigram overlap
    pred_unigrams = set(pred_tokens)
    ref_unigrams = set(ref_tokens)

    if len(ref_unigrams) == 0:
        rouge_1 = 0.0
    else:
        overlap = len(pred_unigrams & ref_unigrams)
        rouge_1 = overlap / len(ref_unigrams)

    # ROUGE-2: Bigram overlap
    def get_bigrams(tokens):
        return set(tuple(tokens[i:i+2]) for i in range(len(tokens)-1))

    pred_bigrams = get_bigrams(pred_tokens)
    ref_bigrams = get_bigrams(ref_tokens)

    if len(ref_bigrams) == 0:
        rouge_2 = 0.0
    else:
        overlap = len(pred_bigrams & ref_bigrams)
        rouge_2 = overlap / len(ref_bigrams)

    # ROUGE-L: Longest Common Subsequence (simplified)
    # Using token-level LCS
    def lcs_length(seq1, seq2):
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    lcs_len = lcs_length(pred_tokens, ref_tokens)

    if len(ref_tokens) == 0:
        rouge_l = 0.0
    else:
        rouge_l = lcs_len / len(ref_tokens)

    return {
        'rouge-1': rouge_1,
        'rouge-2': rouge_2,
        'rouge-l': rouge_l
    }

def evaluate_model(jsonl_path: str, model_name: str) -> Dict:
    """Evaluate ROUGE scores for a single model"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"File: {jsonl_path}")
    print(f"{'='*60}")

    data = load_jsonl(jsonl_path)

    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    for item in data:
        prediction = item.get('answer', '')
        reference = item.get('gt_answer', '')

        if not prediction or not reference:
            continue

        scores = compute_rouge_basic(prediction, reference)
        rouge_1_scores.append(scores['rouge-1'])
        rouge_2_scores.append(scores['rouge-2'])
        rouge_l_scores.append(scores['rouge-l'])

    n_samples = len(rouge_1_scores)

    if n_samples == 0:
        print("⚠️  No valid samples found!")
        return None

    avg_rouge_1 = sum(rouge_1_scores) / n_samples
    avg_rouge_2 = sum(rouge_2_scores) / n_samples
    avg_rouge_l = sum(rouge_l_scores) / n_samples

    print(f"\nSamples evaluated: {n_samples}")
    print(f"ROUGE-1: {avg_rouge_1:.4f}")
    print(f"ROUGE-2: {avg_rouge_2:.4f}")
    print(f"ROUGE-L: {avg_rouge_l:.4f}")

    return {
        'model': model_name,
        'n_samples': n_samples,
        'rouge-1': avg_rouge_1,
        'rouge-2': avg_rouge_2,
        'rouge-l': avg_rouge_l
    }

def main():
    # Base path
    base_path = Path("/home/wayneleo8/SafeDelta/llama2/utility_evaluation/sum/data/gen_answers")

    # Models to evaluate
    models = [
        ("sum_deltaone-v2-rho0.05_vllm.jsonl", "DeltaOne++ (ρ=0.05)"),
        ("sum_vllm-purebad100-3b-DeltaOne-fast-s0.11.jsonl", "DeltaOne-fast"),
        ("sum_vllm-purebad100-3b-SafeDelta-s0.11.jsonl", "SafeDelta"),
    ]

    results = []

    for filename, model_name in models:
        file_path = base_path / filename
        if file_path.exists():
            result = evaluate_model(str(file_path), model_name)
            if result:
                results.append(result)
        else:
            print(f"\n⚠️  File not found: {file_path}")

    # Create results DataFrame
    if results:
        df = pd.DataFrame(results)

        print("\n" + "="*60)
        print("ROUGE EVALUATION SUMMARY")
        print("="*60)
        print(df.to_string(index=False))

        # Save results
        output_dir = Path("/home/wayneleo8/SafeDelta/DeltaOne/experiments/results/utility_evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "rouge_scores.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Results saved to: {csv_path}")

        # Generate LaTeX table
        latex_table = df.to_latex(
            index=False,
            float_format="%.4f",
            column_format='lcccc'
        )

        latex_path = output_dir / "table_rouge.tex"
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"✅ LaTeX table saved to: {latex_path}")

        # Generate comparison visualization
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            fig, ax = plt.subplots(figsize=(10, 6))

            x = np.arange(len(df))
            width = 0.25

            bars1 = ax.bar(x - width, df['rouge-1'], width, label='ROUGE-1', color='#2E86AB')
            bars2 = ax.bar(x, df['rouge-2'], width, label='ROUGE-2', color='#A23B72')
            bars3 = ax.bar(x + width, df['rouge-l'], width, label='ROUGE-L', color='#F18F01')

            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel('ROUGE Score', fontsize=12)
            ax.set_title('ROUGE Scores Comparison (Utility Evaluation)', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(df['model'], rotation=15, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()

            png_path = output_dir / "fig_rouge_comparison.png"
            pdf_path = output_dir / "fig_rouge_comparison.pdf"

            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Figure saved to: {png_path}, {pdf_path}")
        except Exception as e:
            print(f"⚠️  Could not generate figure: {e}")
    else:
        print("\n❌ No results to save!")

if __name__ == "__main__":
    main()
