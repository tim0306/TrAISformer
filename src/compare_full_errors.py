#!/usr/bin/env python
# coding: utf-8
"""
Detailed comparison of full prediction error curves between baseline and Phase 1 models.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load results
with open('baselines/baseline_results.json') as f:
    baseline = json.load(f)

with open('phase1_results.json') as f:
    phase1 = json.load(f)

baseline_errors = np.array(baseline['prediction_errors_full'])
phase1_errors = np.array(phase1['prediction_errors_full'])

# Calculate improvements at each time step
improvements = ((baseline_errors - phase1_errors) / baseline_errors) * 100
absolute_diff = baseline_errors - phase1_errors

# Time steps (10-minute intervals)
time_steps = np.arange(len(baseline_errors)) / 6  # Convert to hours

# Create comprehensive comparison plot
fig = plt.figure(figsize=(16, 10))

# Plot 1: Error curves
ax1 = plt.subplot(2, 2, 1)
ax1.plot(time_steps, baseline_errors, linewidth=2.5, alpha=0.8, color='#FF6B6B',
         label='Baseline', marker='o', markersize=4)
ax1.plot(time_steps, phase1_errors, linewidth=2.5, alpha=0.8, color='#4ECDC4',
         label='Phase 1', marker='s', markersize=4)
ax1.set_xlabel('Prediction Horizon (hours)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Prediction Error (km)', fontsize=12, fontweight='bold')
ax1.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(alpha=0.3, linestyle='--')
ax1.set_xlim([0, max(time_steps)])

# Highlight key horizons
for hour, label in [(1/6, '10min'), (1, '1h'), (2, '2h'), (3, '3h')]:
    ax1.axvline(x=hour, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax1.text(hour, ax1.get_ylim()[1] * 0.95, label, ha='center',
             fontsize=9, color='gray', fontweight='bold')

# Plot 2: Improvement percentage at each time step
ax2 = plt.subplot(2, 2, 2)
colors = ['#4ECDC4' if x > 0 else '#FF6B6B' for x in improvements]
bars = ax2.bar(time_steps, improvements, width=0.15, alpha=0.8, color=colors, edgecolor='black', linewidth=0.5)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Prediction Horizon (hours)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
ax2.set_title('Accuracy Improvement at Each Time Step', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_xlim([0, max(time_steps)])

# Add average line
avg_improvement = np.mean(improvements)
ax2.axhline(y=avg_improvement, color='darkblue', linestyle='--', linewidth=2, alpha=0.7,
            label=f'Average: {avg_improvement:+.2f}%')
ax2.legend(fontsize=10)

# Plot 3: Absolute error difference
ax3 = plt.subplot(2, 2, 3)
colors = ['#4ECDC4' if x > 0 else '#FF6B6B' for x in absolute_diff]
bars = ax3.bar(time_steps, absolute_diff, width=0.15, alpha=0.8, color=colors, edgecolor='black', linewidth=0.5)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_xlabel('Prediction Horizon (hours)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Absolute Error Reduction (km)', fontsize=12, fontweight='bold')
ax3.set_title('Error Reduction (Baseline - Phase 1)', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.set_xlim([0, max(time_steps)])

# Plot 4: Detailed statistics table
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

# Calculate statistics
stats_data = []
horizons_idx = [0, 5, 11, 17]  # 10min, 1h, 2h, 3h
horizon_names = ['10min', '1hour', '2hour', '3hour']

for idx, name in zip(horizons_idx, horizon_names):
    stats_data.append([
        name,
        f"{baseline_errors[idx]:.4f}",
        f"{phase1_errors[idx]:.4f}",
        f"{absolute_diff[idx]:+.4f}",
        f"{improvements[idx]:+.2f}%"
    ])

# Add overall statistics
stats_data.append(['', '', '', '', ''])
stats_data.append([
    'Average',
    f"{np.mean(baseline_errors):.4f}",
    f"{np.mean(phase1_errors):.4f}",
    f"{np.mean(absolute_diff):+.4f}",
    f"{avg_improvement:+.2f}%"
])
stats_data.append([
    'Max Improvement',
    '-',
    '-',
    f"{np.max(absolute_diff):+.4f}",
    f"{np.max(improvements):+.2f}%"
])
stats_data.append([
    'Min (Regression)',
    '-',
    '-',
    f"{np.min(absolute_diff):+.4f}",
    f"{np.min(improvements):+.2f}%"
])

# Create table
table = ax4.table(
    cellText=stats_data,
    colLabels=['Horizon', 'Baseline (km)', 'Phase 1 (km)', 'Δ Error (km)', 'Improvement'],
    cellLoc='center',
    loc='center',
    colWidths=[0.15, 0.18, 0.18, 0.18, 0.18]
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style key rows
for i in range(1, 5):
    for j in range(5):
        table[(i, j)].set_facecolor('#ecf0f1')

# Style summary rows
for i in range(6, 9):
    for j in range(5):
        table[(i, j)].set_facecolor('#d5dbdb')
        table[(i, j)].set_text_props(weight='bold')

ax4.set_title('Detailed Comparison Statistics', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('comparison_results/phase1/full_error_comparison.png', dpi=200, bbox_inches='tight')
print("Full error comparison saved to comparison_results/phase1/full_error_comparison.png")

# Print detailed console output
print("\n" + "="*80)
print("FULL PREDICTION ERROR CURVE COMPARISON")
print("="*80)
print(f"\nTotal time steps analyzed: {len(baseline_errors)} (0 to {max(time_steps):.1f} hours)")
print(f"Interval: 10 minutes per step")
print("\n" + "-"*80)
print("PER-TIMESTEP ANALYSIS:")
print("-"*80)
print(f"{'Time':>8s} {'Baseline':>12s} {'Phase 1':>12s} {'Δ Error':>12s} {'Improvement':>12s}")
print("-"*80)

for i, t in enumerate(time_steps):
    hours = int(t)
    mins = int((t - hours) * 60)
    time_str = f"{hours}h{mins:02d}m"
    print(f"{time_str:>8s} {baseline_errors[i]:>11.4f}km {phase1_errors[i]:>11.4f}km "
          f"{absolute_diff[i]:>+11.4f}km {improvements[i]:>+11.2f}%")

print("-"*80)
print(f"{'Average':>8s} {np.mean(baseline_errors):>11.4f}km {np.mean(phase1_errors):>11.4f}km "
      f"{np.mean(absolute_diff):>+11.4f}km {avg_improvement:>+11.2f}%")
print("="*80)

# Summary statistics
print("\nSUMMARY STATISTICS:")
print("-"*80)
print(f"Best improvement:     {np.max(improvements):+.2f}% at {time_steps[np.argmax(improvements)]:.2f}h "
      f"({absolute_diff[np.argmax(improvements)]:+.4f} km)")
print(f"Worst (regression):   {np.min(improvements):+.2f}% at {time_steps[np.argmin(improvements)]:.2f}h "
      f"({absolute_diff[np.argmin(improvements)]:+.4f} km)")
print(f"Median improvement:   {np.median(improvements):+.2f}%")
print(f"Std dev improvement:  {np.std(improvements):.2f}%")
print(f"\nTime steps with improvement: {np.sum(improvements > 0)}/{len(improvements)} "
      f"({np.sum(improvements > 0)/len(improvements)*100:.1f}%)")
print(f"Time steps with regression:  {np.sum(improvements < 0)}/{len(improvements)} "
      f"({np.sum(improvements < 0)/len(improvements)*100:.1f}%)")
print("="*80)

# Save detailed results to JSON
detailed_comparison = {
    'time_steps_hours': time_steps.tolist(),
    'baseline_errors_km': baseline_errors.tolist(),
    'phase1_errors_km': phase1_errors.tolist(),
    'absolute_difference_km': absolute_diff.tolist(),
    'improvement_percent': improvements.tolist(),
    'summary': {
        'average_improvement_percent': float(avg_improvement),
        'median_improvement_percent': float(np.median(improvements)),
        'best_improvement_percent': float(np.max(improvements)),
        'worst_improvement_percent': float(np.min(improvements)),
        'timesteps_with_improvement': int(np.sum(improvements > 0)),
        'timesteps_with_regression': int(np.sum(improvements < 0)),
        'total_timesteps': len(improvements)
    }
}

with open('comparison_results/phase1/full_error_detailed.json', 'w') as f:
    json.dump(detailed_comparison, f, indent=2)
print("\nDetailed comparison data saved to comparison_results/phase1/full_error_detailed.json")
