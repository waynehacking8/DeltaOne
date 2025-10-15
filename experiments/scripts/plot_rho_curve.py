#!/usr/bin/env python3
"""
Experiment C: ρ vs ASR Curve Visualization

Plots the relationship between selection ratio (ρ) and Attack Success Rate (ASR)
to identify the optimal sweet spot (expected around ρ ≈ 0.10-0.12)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse


# SCI-quality plot settings
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2.0,
    'patch.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# Colorblind-friendly palette
COLORS = sns.color_palette("colorblind")


def plot_rho_vs_asr(df: pd.DataFrame, output_dir: str):
    """
    Plot ASR vs ρ curve with optimal point marked

    Expected: U-shaped curve with minimum around ρ ≈ 0.10-0.12
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Filter for ρ sweep data (DeltaOne++ or similar methods with rho field)
    df_rho = df[df['rho'].notna()].copy()
    df_rho['rho_val'] = df_rho['rho'].astype(float)
    df_rho = df_rho.sort_values('rho_val')

    if len(df_rho) < 2:
        print("❌ Not enough ρ sweep data for plotting")
        print(f"   Found {len(df_rho)} data points, need at least 2")
        return

    print(f"📊 Plotting ρ vs ASR curve with {len(df_rho)} data points")

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot line with markers
    ax.plot(df_rho['rho_val'], df_rho['asr'],
            marker='o', markersize=10, linewidth=2.5,
            color=COLORS[2], label='DeltaOne++',
            markerfacecolor=COLORS[2], markeredgecolor='black',
            markeredgewidth=1.0)

    # Mark optimal point (minimum ASR)
    min_idx = df_rho['asr'].idxmin()
    optimal_rho = df_rho.loc[min_idx, 'rho_val']
    optimal_asr = df_rho.loc[min_idx, 'asr']

    ax.scatter([optimal_rho], [optimal_asr],
              s=400, marker='*', color='red',
              edgecolor='black', linewidth=2.0,
              label=f'Optimal (ρ={optimal_rho:.2f}, ASR={optimal_asr:.1f}%)',
              zorder=5)

    # Add vertical line at optimal ρ
    ax.axvline(x=optimal_rho, color='red', linestyle='--',
               linewidth=1.5, alpha=0.5)

    # Add SafeDelta baseline if available
    df_safedelta = df[df['method'] == 'SafeDelta']
    if not df_safedelta.empty:
        safedelta_asr = df_safedelta.iloc[0]['asr']
        ax.axhline(y=safedelta_asr, color='blue', linestyle='--',
                   linewidth=1.5, alpha=0.5,
                   label=f'SafeDelta baseline ({safedelta_asr:.1f}%)')

    # Styling
    ax.set_xlabel('Selection Ratio (ρ)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=12)
    ax.set_title('Safety vs Parameter Selection Ratio',
                 fontweight='bold', pad=15, fontsize=13)

    # Set axis limits
    ax.set_xlim(df_rho['rho_val'].min() - 0.02, df_rho['rho_val'].max() + 0.02)
    ax.set_ylim(0, max(df_rho['asr'].max() * 1.15, 25))

    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(frameon=True, loc='best', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add annotation box
    textstr = (
        f"Optimal ρ: {optimal_rho:.2f}\n"
        f"Range tested: {df_rho['rho_val'].min():.2f} - {df_rho['rho_val'].max():.2f}\n"
        f"ASR reduction: {df_rho['asr'].max() - optimal_asr:.1f}%"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', bbox=props)

    # Save
    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        save_path = output_path / f'fig_rho_vs_asr.{fmt}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")

    plt.close()

    # Print statistics
    print("\n" + "="*60)
    print("ρ vs ASR Analysis")
    print("="*60)
    print(df_rho[['rho_val', 'asr', 'method']].to_string(index=False))
    print(f"\nOptimal ρ: {optimal_rho:.3f}")
    print(f"Optimal ASR: {optimal_asr:.2f}%")
    print(f"ASR range: {df_rho['asr'].min():.2f}% - {df_rho['asr'].max():.2f}%")


def plot_rho_convergence(sweep_summary_path: str, output_dir: str):
    """
    Plot ρ-targeting convergence: target ρ vs actual ρ

    Shows how well ρ-targeting control works
    """
    import json

    output_path = Path(output_dir)

    # Load sweep summary
    with open(sweep_summary_path, 'r') as f:
        summary = json.load(f)

    # Extract target vs actual ρ
    data = []
    for result in summary['results']:
        rho_target = result['rho']
        stats_path = result.get('stats_path')

        if stats_path and Path(stats_path).exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            rho_actual = stats.get('selection_ratio', rho_target)
        else:
            rho_actual = rho_target  # Fallback

        data.append({'target': rho_target, 'actual': rho_actual})

    df_conv = pd.DataFrame(data)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot diagonal (perfect convergence)
    lims = [0, max(df_conv['target'].max(), df_conv['actual'].max()) * 1.05]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5, label='Perfect convergence')

    # Plot actual points
    ax.scatter(df_conv['target'], df_conv['actual'],
              s=100, alpha=0.7, color=COLORS[2],
              edgecolor='black', linewidth=1.0)

    # Add error bars
    errors = (df_conv['actual'] - df_conv['target']).abs()
    for _, row in df_conv.iterrows():
        ax.plot([row['target'], row['target']],
               [row['target'], row['actual']],
               'r-', alpha=0.3, linewidth=1.0)

    # Styling
    ax.set_xlabel('Target ρ', fontweight='bold', fontsize=11)
    ax.set_ylabel('Actual ρ', fontweight='bold', fontsize=11)
    ax.set_title('ρ-Targeting Convergence', fontweight='bold', pad=15, fontsize=12)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(frameon=True, loc='upper left')
    ax.set_aspect('equal', 'box')

    # Calculate MAE
    mae = errors.mean()
    ax.text(0.95, 0.05, f'MAE: {mae:.4f}',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Save
    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        save_path = output_path / f'fig_rho_convergence.{fmt}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")

    plt.close()

    print(f"\n✅ ρ-Targeting MAE: {mae:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot ρ vs ASR curve (Experiment C)')

    parser.add_argument('--csv', type=str,
                       default='/home/wayneleo8/SafeDelta/DeltaOne/experiments/results/asr_analysis/asr_results.csv',
                       help='Path to ASR results CSV')
    parser.add_argument('--sweep-summary', type=str,
                       default='/home/wayneleo8/SafeDelta/DeltaOne/experiments/results/exp_c_rho_sweep/sweep_summary.json',
                       help='Path to sweep summary JSON')
    parser.add_argument('--output-dir', type=str,
                       default='/home/wayneleo8/SafeDelta/DeltaOne/experiments/results/exp_c_rho_sweep',
                       help='Output directory for figures')

    args = parser.parse_args()

    print("="*60)
    print("Experiment C: ρ Curve Visualization")
    print("="*60)

    # Load ASR results
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} ASR results")

    # Plot ρ vs ASR curve
    plot_rho_vs_asr(df, args.output_dir)

    # Plot convergence if sweep summary exists
    if Path(args.sweep_summary).exists():
        plot_rho_convergence(args.sweep_summary, args.output_dir)
    else:
        print(f"\n⚠️  Sweep summary not found: {args.sweep_summary}")
        print("   Skipping convergence plot")

    print("\n✅ ρ curve visualization complete!")
    print(f"   Results: {args.output_dir}")


if __name__ == '__main__':
    main()
