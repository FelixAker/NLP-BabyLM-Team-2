#!/usr/bin/env python3
"""
Visualization script for token imbalance analysis results.

Generates three key visualizations:
1. Error rate vs |Δtokens| plot
2. Per-category sensitivity table
3. Bias reduction evidence plot
"""

import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='token_imbalance_results.json')
parser.add_argument('--output_prefix', type=str, default='token_imbalance')

def plot_error_vs_delta(data, output_prefix):
    """Plot error rate as a function of |Δtokens|."""
    
    binned_accuracies = data['overall_statistics']['binned_accuracies']
    metrics = ['raw_log_prob', 'normalized_log_prob', 'bpc', 'bpb']
    metric_labels = {
        'raw_log_prob': 'Raw Log-Prob',
        'normalized_log_prob': 'Per-Token Normalized',
        'bpc': 'Bits-per-Character',
        'bpb': 'Bits-per-Byte'
    }
    
    # Prepare data for plotting
    bins = sorted(binned_accuracies.keys(), key=lambda x: 0 if x == '0' else (int(x) if x != '4+' else 5))
    
    plt.figure(figsize=(10, 6))
    
    for metric in metrics:
        error_rates = []
        counts = []
        
        for bin_label in bins:
            if bin_label in binned_accuracies:
                error_rates.append(binned_accuracies[bin_label][metric]['error_rate'])
                counts.append(binned_accuracies[bin_label][metric]['count'])
            else:
                error_rates.append(0)
                counts.append(0)
        
        plt.plot(bins, error_rates, marker='o', linewidth=2, markersize=8, label=metric_labels[metric])
    
    plt.xlabel('|Δtokens| (Token Count Difference)', fontsize=12, fontweight='bold')
    plt.ylabel('Error Rate', fontsize=12, fontweight='bold')
    plt.title('Model Error Rate vs Token Count Imbalance', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = f'{output_prefix}_error_vs_delta.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_category_table(data, output_prefix):
    """Create a table visualization of per-category correlations."""
    
    category_stats = data['category_statistics']
    metrics = ['raw_log_prob', 'normalized_log_prob', 'bpc', 'bpb']
    metric_labels = {
        'raw_log_prob': 'Raw\nLog-Prob',
        'normalized_log_prob': 'Normalized\nLog-Prob',
        'bpc': 'Bits-per-\nCharacter',
        'bpb': 'Bits-per-\nByte'
    }
    
    # Prepare data
    categories = list(category_stats.keys())
    
    # Create DataFrame
    table_data = []
    for category in categories:
        row = {'Category': category, 'Sample Size': category_stats[category]['sample_size']}
        for metric in metrics:
            corr = category_stats[category]['correlations'][metric]['correlation']
            row[metric_labels[metric]] = f"{corr:.3f}"
        table_data.append(row)
    
    # Add overall statistics
    overall_corr = data['overall_statistics']['correlations']
    overall_row = {'Category': 'Overall', 'Sample Size': data['total_pairs']}
    for metric in metrics:
        corr = overall_corr[metric]['correlation']
        overall_row[metric_labels[metric]] = f"{corr:.3f}"
    table_data.append(overall_row)
    
    df = pd.DataFrame(table_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                     cellLoc='center', loc='center',
                     colWidths=[0.15, 0.12, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            cell = table[(i, j)]
            if i == len(df):  # Overall row
                cell.set_facecolor('#E7E6E6')
                cell.set_text_props(weight='bold')
            elif i % 2 == 0:
                cell.set_facecolor('#F2F2F2')
    
    plt.title('Correlation Between |Δtokens| and Prediction Errors by Category', 
              fontsize=14, fontweight='bold', pad=20)
    
    output_file = f'{output_prefix}_category_table.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_bias_reduction(data, output_prefix):
    """Plot evidence that normalization reduces token count bias."""
    
    pairs = data['pairs']
    
    # Sample data for visualization (use subset for clarity)
    np.random.seed(42)
    sample_size = min(2000, len(pairs))
    sample_indices = np.random.choice(len(pairs), sample_size, replace=False)
    
    abs_deltas = []
    raw_errors = []
    norm_errors = []
    
    for idx in sample_indices:
        pair = pairs[idx]
        abs_deltas.append(pair['abs_delta_tokens'])
        raw_errors.append(0 if pair['predictions']['raw_log_prob']['correct'] else 1)
        norm_errors.append(0 if pair['predictions']['normalized_log_prob']['correct'] else 1)
    
    abs_deltas = np.array(abs_deltas)
    raw_errors = np.array(raw_errors)
    norm_errors = np.array(norm_errors)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Raw Log-Prob
    # Bin by delta and compute error rate
    bins = range(0, max(abs_deltas) + 2)
    raw_error_rates = []
    norm_error_rates = []
    bin_centers = []
    
    for i in range(len(bins) - 1):
        mask = (abs_deltas >= bins[i]) & (abs_deltas < bins[i + 1])
        if mask.sum() > 0:
            raw_error_rates.append(raw_errors[mask].mean())
            norm_error_rates.append(norm_errors[mask].mean())
            bin_centers.append(bins[i])
    
    # Raw correlation
    from scipy import stats
    raw_corr, raw_p = stats.pearsonr(abs_deltas, raw_errors)
    norm_corr, norm_p = stats.pearsonr(abs_deltas, norm_errors)
    
    ax1.scatter(bin_centers, raw_error_rates, alpha=0.6, s=100, color='#E74C3C', label='Raw Log-Prob')
    if len(bin_centers) > 1:
        z = np.polyfit(bin_centers, raw_error_rates, 1)
        p = np.poly1d(z)
        ax1.plot(bin_centers, p(bin_centers), "--", color='#E74C3C', alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('|Δtokens|', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Error Rate', fontsize=11, fontweight='bold')
    ax1.set_title(f'Raw Log-Prob\n(r = {raw_corr:.3f}, p = {raw_p:.3e})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Normalized Log-Prob
    ax2.scatter(bin_centers, norm_error_rates, alpha=0.6, s=100, color='#3498DB', label='Normalized Log-Prob')
    if len(bin_centers) > 1:
        z = np.polyfit(bin_centers, norm_error_rates, 1)
        p = np.poly1d(z)
        ax2.plot(bin_centers, p(bin_centers), "--", color='#3498DB', alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('|Δtokens|', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Error Rate', fontsize=11, fontweight='bold')
    ax2.set_title(f'Per-Token Normalized Log-Prob\n(r = {norm_corr:.3f}, p = {norm_p:.3e})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.suptitle('Evidence of Bias Reduction Through Normalization', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = f'{output_prefix}_bias_reduction.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    args = parser.parse_args()
    
    print(f"Loading results from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    print("\nGenerating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Generate plots
    plot_error_vs_delta(data, args.output_prefix)
    plot_category_table(data, args.output_prefix)
    plot_bias_reduction(data, args.output_prefix)
    
    print("\n✓ All visualizations generated successfully!")


if __name__ == '__main__':
    main()
