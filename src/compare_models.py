#!/usr/bin/env python
# coding: utf-8
"""
Compare two TrAISformer models side-by-side on identical test data.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os


def load_results(path):
    """Load evaluation results from JSON file."""
    with open(path) as f:
        return json.load(f)


def compare_models(baseline_path, optimized_path, output_dir='comparison_results'):
    """
    Compare baseline and optimized models.

    Args:
        baseline_path: Path to baseline evaluation JSON
        optimized_path: Path to optimized evaluation JSON
        output_dir: Directory to save comparison results

    Returns:
        dict: Comparison report
    """
    # Load results
    print("Loading evaluation results...")
    baseline = load_results(baseline_path)
    optimized = load_results(optimized_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Calculate improvements
    improvements = {}
    for horizon in ['10min', '1hour', '2hour', '3hour']:
        baseline_err = baseline['prediction_errors_km'].get(horizon)
        optimized_err = optimized['prediction_errors_km'].get(horizon)

        if baseline_err is not None and optimized_err is not None:
            improvement_pct = (baseline_err - optimized_err) / baseline_err * 100

            improvements[horizon] = {
                'baseline_km': baseline_err,
                'optimized_km': optimized_err,
                'improvement_percent': improvement_pct,
                'absolute_improvement_km': baseline_err - optimized_err
            }

    # Plot comparison
    fig = plt.figure(figsize=(14, 6))

    # Subplot 1: Prediction errors at different horizons
    ax1 = plt.subplot(1, 3, 1)
    horizons = list(improvements.keys())
    baseline_errs = [improvements[h]['baseline_km'] for h in horizons]
    optimized_errs = [improvements[h]['optimized_km'] for h in horizons]

    x = np.arange(len(horizons))
    width = 0.35
    bars1 = ax1.bar(x - width/2, baseline_errs, width, label='Baseline', alpha=0.8, color='#FF6B6B')
    bars2 = ax1.bar(x + width/2, optimized_errs, width, label='Optimized', alpha=0.8, color='#4ECDC4')

    ax1.set_xlabel('Prediction Horizon', fontsize=11)
    ax1.set_ylabel('Mean Error (km)', fontsize=11)
    ax1.set_title('Prediction Error Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)

    # Subplot 2: Improvement percentages
    ax2 = plt.subplot(1, 3, 2)
    improvement_pcts = [improvements[h]['improvement_percent'] for h in horizons]
    colors = ['#4ECDC4' if x > 0 else '#FF6B6B' for x in improvement_pcts]
    bars = ax2.bar(horizons, improvement_pcts, alpha=0.8, color=colors)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Prediction Horizon', fontsize=11)
    ax2.set_ylabel('Improvement (%)', fontsize=11)
    ax2.set_title('Accuracy Improvement', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, improvement_pcts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.1f}%',
                ha='center', va='bottom' if val > 0 else 'top', fontsize=9)

    # Subplot 3: Inference speed comparison
    ax3 = plt.subplot(1, 3, 3)
    baseline_time = baseline['inference_time_ms']
    optimized_time = optimized['inference_time_ms']
    speedup = baseline_time / optimized_time if optimized_time > 0 else 0

    bars = ax3.bar(['Baseline', 'Optimized'], [baseline_time, optimized_time],
                   alpha=0.8, color=['#FF6B6B', '#4ECDC4'])
    ax3.set_ylabel('Inference Time (ms)', fontsize=11)
    ax3.set_title(f'Inference Speed\n(Speedup: {speedup:.2f}×)', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}ms',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    comparison_plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {comparison_plot_path}")
    plt.close()

    # Plot full error curves
    fig = plt.figure(figsize=(10, 6))
    baseline_curve = baseline.get('prediction_errors_full', [])
    optimized_curve = optimized.get('prediction_errors_full', [])

    if baseline_curve and optimized_curve:
        time_steps = np.arange(len(baseline_curve)) / 6  # Convert to hours (assuming 10min intervals)
        plt.plot(time_steps, baseline_curve, label='Baseline', linewidth=2, alpha=0.8, color='#FF6B6B')
        plt.plot(time_steps, optimized_curve, label='Optimized', linewidth=2, alpha=0.8, color='#4ECDC4')

        plt.xlabel('Time (hours)', fontsize=12)
        plt.ylabel('Prediction Error (km)', fontsize=12)
        plt.title('Prediction Error Over Time', fontsize=13, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.xlim([0, max(time_steps)])

        error_curve_path = os.path.join(output_dir, 'error_curves.png')
        plt.savefig(error_curve_path, dpi=150, bbox_inches='tight')
        print(f"Error curves saved to {error_curve_path}")
        plt.close()

    # Print detailed summary
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    print(f"\nBaseline Model: {baseline.get('model_path', 'N/A')}")
    print(f"Optimized Model: {optimized.get('model_path', 'N/A')}")
    print(f"\nTest Samples: {baseline.get('total_test_samples', 'N/A')}")

    print("\n" + "-"*60)
    print("PREDICTION ACCURACY:")
    print("-"*60)
    for horizon, data in improvements.items():
        print(f"\n{horizon}:")
        print(f"  Baseline:  {data['baseline_km']:.4f} km")
        print(f"  Optimized: {data['optimized_km']:.4f} km")
        print(f"  Δ Error:   {data['absolute_improvement_km']:+.4f} km")
        print(f"  Improvement: {data['improvement_percent']:+.2f}%")

    # Average improvement
    avg_improvement = np.mean([v['improvement_percent'] for v in improvements.values()])
    print(f"\nAverage Improvement: {avg_improvement:+.2f}%")

    print("\n" + "-"*60)
    print("INFERENCE SPEED:")
    print("-"*60)
    print(f"Baseline:  {baseline_time:.2f} ± {baseline.get('inference_time_std_ms', 0):.2f} ms")
    print(f"Optimized: {optimized_time:.2f} ± {optimized.get('inference_time_std_ms', 0):.2f} ms")
    print(f"Speedup: {speedup:.2f}×")

    if speedup > 1:
        print(f"⚡ {(speedup - 1) * 100:.1f}% faster inference")
    elif speedup < 1:
        print(f"⚠️  {(1 - speedup) * 100:.1f}% slower inference")

    print("="*60)

    # Create summary
    summary = f"Optimized model is {speedup:.2f}× faster with {avg_improvement:+.1f}% average accuracy change"
    if avg_improvement > 0:
        summary += f" ✅ IMPROVEMENT"
    elif avg_improvement < 0:
        summary += f" ⚠️  REGRESSION"

    # Save comparison report
    report = {
        'baseline_model': baseline.get('model_path', 'N/A'),
        'optimized_model': optimized.get('model_path', 'N/A'),
        'improvements': improvements,
        'inference_speedup': speedup,
        'average_accuracy_improvement_percent': avg_improvement,
        'summary': summary
    }

    report_path = os.path.join(output_dir, 'comparison_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nComparison report saved to {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description='Compare two TrAISformer models')
    parser.add_argument('--baseline', type=str, default='baselines/baseline_results.json',
                        help='Path to baseline evaluation results JSON')
    parser.add_argument('--optimized', type=str, required=True,
                        help='Path to optimized evaluation results JSON')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                        help='Directory to save comparison results')
    args = parser.parse_args()

    compare_models(args.baseline, args.optimized, args.output_dir)


if __name__ == "__main__":
    main()
