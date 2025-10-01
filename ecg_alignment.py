"""
ECG Peak Matching Script
Finds where Holter recording starts in EDF recording by comparing peak patterns.

Method: Compares time intervals from the first peak to all subsequent peaks.
For each potential starting position in EDF, calculates intervals from that peak
to subsequent peaks and matches against Holter's interval pattern.

Usage:
    python script.py --edf path/to/EDF.csv --holter path/to/holter.csv
    python script.py -e EDF.csv -H holter.csv -o results.png
"""

import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import argparse
import sys

def calculate_intervals_from_first(peak_times):
    """Calculate time intervals from the first peak to all subsequent peaks"""
    return peak_times - peak_times[0]

def find_exact_match(edf_peak_times, holter_intervals_from_first, num_holter_peaks, tolerance=0.01):
    """
    Try to find exact matches within a tolerance
    tolerance: acceptable difference in seconds
    """
    best_matches = []
    
    # Try each EDF peak as potential start (up to last - num_holter_peaks)
    for i in range(len(edf_peak_times) - num_holter_peaks + 1):
        # Calculate intervals from this potential start peak to subsequent peaks
        edf_segment_times = edf_peak_times[i:i + num_holter_peaks]
        edf_intervals_from_first = edf_segment_times - edf_segment_times[0]
        
        # Compare with holter intervals
        differences = np.abs(edf_intervals_from_first - holter_intervals_from_first)
        
        # Check if all differences are within tolerance
        if np.all(differences < tolerance):
            max_diff = np.max(differences)
            mean_diff = np.mean(differences)
            best_matches.append({
                'position': i,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'match_quality': 'EXACT'
            })
    
    return best_matches

def find_best_match_correlation(edf_peak_times, holter_intervals_from_first, num_holter_peaks):
    """
    Find best match using correlation
    """
    best_position = -1
    best_correlation = -np.inf
    correlation_scores = []
    
    for i in range(len(edf_peak_times) - num_holter_peaks + 1):
        # Calculate intervals from this potential start peak
        edf_segment_times = edf_peak_times[i:i + num_holter_peaks]
        edf_intervals_from_first = edf_segment_times - edf_segment_times[0]
        
        # Calculate correlation coefficient
        corr = np.corrcoef(edf_intervals_from_first, holter_intervals_from_first)[0, 1]
        correlation_scores.append(corr)
        
        if corr > best_correlation:
            best_correlation = corr
            best_position = i
    
    return best_position, best_correlation, correlation_scores

def find_best_match_mse(edf_peak_times, holter_intervals_from_first, num_holter_peaks):
    """
    Find best match using Mean Squared Error
    Lower MSE means better match
    """
    mse_scores = []
    
    for i in range(len(edf_peak_times) - num_holter_peaks + 1):
        # Calculate intervals from this potential start peak
        edf_segment_times = edf_peak_times[i:i + num_holter_peaks]
        edf_intervals_from_first = edf_segment_times - edf_segment_times[0]
        
        # Calculate MSE
        mse = np.mean((edf_intervals_from_first - holter_intervals_from_first) ** 2)
        mse_scores.append(mse)
    
    best_position = np.argmin(mse_scores)
    best_mse = mse_scores[best_position]
    
    return best_position, best_mse, mse_scores

def convert_to_sample_number(peak_position, edf_sampling_rate=250, holter_sampling_rate=1000):
    """
    Convert peak position to sample numbers accounting for different sampling rates
    """
    ratio = holter_sampling_rate / edf_sampling_rate
    return int(peak_position * ratio)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Find where Holter recording starts in EDF recording by comparing ECG peak patterns.')
parser.add_argument('--edf', '-e', required=True, help='Path to EDF CSV file')
parser.add_argument('--holter', '-H', required=True, help='Path to Holter CSV file')
parser.add_argument('--output', '-o', default='ecg_matching_results.png', help='Output plot filename (default: ecg_matching_results.png)')

args = parser.parse_args()

# Load the CSV files
print("Loading CSV files...")
print(f"EDF file: {args.edf}")
print(f"Holter file: {args.holter}")

try:
    edf_df = pd.read_csv(args.edf)
    holter_df = pd.read_csv(args.holter)
except FileNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading files: {e}")
    sys.exit(1)

print(f"EDF peaks: {len(edf_df)}")
print(f"Holter peaks: {len(holter_df)}")

# Extract peak times
edf_peak_times = edf_df['peak_time_sec'].values
holter_peak_times = holter_df['peak_time_sec'].values

# Calculate intervals from first peak
holter_intervals_from_first = calculate_intervals_from_first(holter_peak_times)
num_holter_peaks = len(holter_peak_times)

print(f"\nEDF peaks: {len(edf_peak_times)}")
print(f"Holter peaks: {num_holter_peaks}")
print(f"Holter recording duration: {holter_intervals_from_first[-1]:.2f} seconds")
print(f"Will test {len(edf_peak_times) - num_holter_peaks + 1} possible starting positions in EDF")

# Step 1: Try exact matches with different tolerances
print("\n" + "="*70)
print("STEP 1: Searching for exact matches...")
print("="*70)

tolerances = [0.005, 0.01, 0.02, 0.05]  # 5ms, 10ms, 20ms, 50ms
exact_matches_found = False

for tol in tolerances:
    matches = find_exact_match(edf_peak_times, holter_intervals_from_first, num_holter_peaks, tolerance=tol)
    if matches:
        exact_matches_found = True
        print(f"\nFound {len(matches)} match(es) with tolerance {tol*1000:.0f}ms:")
        for match in matches:
            edf_peak_idx = match['position']
            edf_start_time = edf_peak_times[edf_peak_idx]
            edf_sample_num = edf_df.iloc[edf_peak_idx]['peak_sample']
            
            print(f"  - EDF peak index: {edf_peak_idx}")
            print(f"    EDF peak sample: {edf_sample_num}")
            print(f"    EDF start time: {edf_start_time:.3f} seconds")
            print(f"    Max difference: {match['max_diff']*1000:.2f}ms")
            print(f"    Mean difference: {match['mean_diff']*1000:.2f}ms")
        break

if not exact_matches_found:
    print("No exact matches found with tested tolerances.")

# Step 2: Find best match using correlation
print("\n" + "="*70)
print("STEP 2: Finding best match using correlation analysis...")
print("="*70)

best_pos_corr, best_corr, corr_scores = find_best_match_correlation(edf_peak_times, holter_intervals_from_first, num_holter_peaks)

edf_start_time_corr = edf_peak_times[best_pos_corr]
edf_sample_corr = edf_df.iloc[best_pos_corr]['peak_sample']

print(f"\nBest match by correlation:")
print(f"  EDF peak index: {best_pos_corr}")
print(f"  EDF peak sample: {edf_sample_corr}")
print(f"  EDF start time: {edf_start_time_corr:.3f} seconds")
print(f"  Correlation coefficient: {best_corr:.6f}")

# Calculate sample number for Holter equivalent (accounting for 4x sampling rate)
holter_equivalent_sample = convert_to_sample_number(best_pos_corr)
print(f"  Holter equivalent sample (at 1000Hz): {holter_equivalent_sample}")

# Step 3: Find best match using MSE
print("\n" + "="*70)
print("STEP 3: Finding best match using Mean Squared Error...")
print("="*70)

best_pos_mse, best_mse, mse_scores = find_best_match_mse(edf_peak_times, holter_intervals_from_first, num_holter_peaks)

edf_start_time_mse = edf_peak_times[best_pos_mse]
edf_sample_mse = edf_df.iloc[best_pos_mse]['peak_sample']

print(f"\nBest match by MSE:")
print(f"  EDF peak index: {best_pos_mse}")
print(f"  EDF peak sample: {edf_sample_mse}")
print(f"  EDF start time: {edf_start_time_mse:.3f} seconds")
print(f"  MSE: {best_mse:.6f}")

# Step 4: Validate the best match
print("\n" + "="*70)
print("STEP 4: Validation of best match...")
print("="*70)

# Use correlation-based match for validation
validation_pos = best_pos_corr
edf_segment_times = edf_peak_times[validation_pos:validation_pos + num_holter_peaks]
edf_intervals_from_first = edf_segment_times - edf_segment_times[0]

# Compare first 10 intervals from first peak
print("\nComparison of time intervals from first peak (first 10):")
print(f"{'Index':<8} {'EDF (s)':<12} {'Holter (s)':<12} {'Diff (ms)':<12}")
print("-" * 50)
for i in range(min(10, num_holter_peaks)):
    diff_ms = (edf_intervals_from_first[i] - holter_intervals_from_first[i]) * 1000
    print(f"{i:<8} {edf_intervals_from_first[i]:<12.6f} {holter_intervals_from_first[i]:<12.6f} {diff_ms:<12.3f}")

# Calculate overall statistics
differences = np.abs(edf_intervals_from_first - holter_intervals_from_first) * 1000  # in ms
print(f"\nOverall statistics (in milliseconds):")
print(f"  Mean difference: {np.mean(differences):.3f}ms")
print(f"  Median difference: {np.median(differences):.3f}ms")
print(f"  Std deviation: {np.std(differences):.3f}ms")
print(f"  Max difference: {np.max(differences):.3f}ms")

# Final summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"\nHolter recording starts at:")
print(f"  EDF Peak Index: {best_pos_corr}")
print(f"  EDF Peak Sample: {edf_sample_corr}")
print(f"  EDF Time: {edf_start_time_corr:.3f} seconds")
print(f"  Confidence: {best_corr:.4f} (correlation coefficient)")

# Plot the results
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Correlation scores across all positions
axes[0].plot(corr_scores, linewidth=1)
axes[0].axvline(best_pos_corr, color='r', linestyle='--', label=f'Best match at {best_pos_corr}')
axes[0].set_xlabel('EDF Peak Index')
axes[0].set_ylabel('Correlation Coefficient')
axes[0].set_title('Correlation Score vs Position')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: MSE scores across all positions
axes[1].plot(mse_scores, linewidth=1, color='orange')
axes[1].axvline(best_pos_mse, color='r', linestyle='--', label=f'Best match at {best_pos_mse}')
axes[1].set_xlabel('EDF Peak Index')
axes[1].set_ylabel('Mean Squared Error')
axes[1].set_title('MSE vs Position')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Interval comparison at best match
comparison_length = min(50, num_holter_peaks)
x_indices = range(comparison_length)
edf_plot = edf_intervals_from_first[:comparison_length]
holter_plot = holter_intervals_from_first[:comparison_length]

axes[2].plot(x_indices, edf_plot, 'b-', label='EDF', linewidth=2, alpha=0.7)
axes[2].plot(x_indices, holter_plot, 'r--', label='Holter', linewidth=2, alpha=0.7)
axes[2].set_xlabel('Peak Index')
axes[2].set_ylabel('Time from First Peak (seconds)')
axes[2].set_title(f'Time Intervals from First Peak - Best Match (Position {best_pos_corr})')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"\nPlot saved as '{args.output}'")
plt.show()

print("\nAnalysis complete!")
