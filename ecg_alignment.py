import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import argparse
import sys

def calculate_intervals(peak_times):
    """Calculate RR intervals from peak times."""
    return np.diff(peak_times)

def normalize_intervals(intervals):
    """Normalize intervals to zero mean and unit variance."""
    return (intervals - np.mean(intervals)) / np.std(intervals)

def calculate_similarity(edf_intervals, holter_intervals, method='correlation'):
    """
    Calculate similarity between two interval sequences.
    
    Methods:
    - 'correlation': Pearson correlation coefficient
    - 'rmse': Root mean squared error (lower is better)
    - 'mae': Mean absolute error (lower is better)
    """
    if len(edf_intervals) != len(holter_intervals):
        raise ValueError("Interval sequences must have same length")
    
    if method == 'correlation':
        # Higher correlation is better
        return np.corrcoef(edf_intervals, holter_intervals)[0, 1]
    elif method == 'rmse':
        # Lower RMSE is better (return negative for consistent "higher is better" logic)
        return -np.sqrt(np.mean((edf_intervals - holter_intervals) ** 2))
    elif method == 'mae':
        # Lower MAE is better (return negative for consistent "higher is better" logic)
        return -np.mean(np.abs(edf_intervals - holter_intervals))
    else:
        raise ValueError(f"Unknown method: {method}")

def find_best_match(edf_intervals, holter_intervals, method='correlation', normalize=True):
    """
    Find the best matching position of holter intervals within edf intervals.
    
    Returns:
        best_position: Index in edf_intervals where holter starts
        scores: Array of similarity scores for each position
        best_score: The best similarity score
    """
    holter_len = len(holter_intervals)
    edf_len = len(edf_intervals)
    
    if holter_len > edf_len:
        raise ValueError("Holter recording is longer than EDF recording")
    
    # Normalize if requested
    if normalize:
        holter_norm = normalize_intervals(holter_intervals)
    else:
        holter_norm = holter_intervals
    
    scores = []
    
    # Slide holter pattern through edf intervals
    for i in range(edf_len - holter_len + 1):
        edf_segment = edf_intervals[i:i + holter_len]
        
        if normalize:
            edf_segment = normalize_intervals(edf_segment)
        
        score = calculate_similarity(edf_segment, holter_norm, method)
        scores.append(score)
    
    scores = np.array(scores)
    best_position = np.argmax(scores)
    best_score = scores[best_position]
    
    return best_position, scores, best_score

def main(edf_path, holter_path):
    # Load the CSV files
    print("Loading CSV files...")
    print(f"EDF file: {edf_path}")
    print(f"Holter file: {holter_path}")
    edf_df = pd.read_csv(edf_path)
    holter_df = pd.read_csv(holter_path)
    
    print(f"EDF peaks: {len(edf_df)}")
    print(f"Holter peaks: {len(holter_df)}")
    
    # Extract peak times
    edf_times = edf_df['peak_time_sec'].values
    holter_times = holter_df['peak_time_sec'].values
    
    # Calculate RR intervals
    print("\nCalculating RR intervals...")
    edf_intervals = calculate_intervals(edf_times)
    holter_intervals = calculate_intervals(holter_times)
    
    print(f"EDF intervals: {len(edf_intervals)}")
    print(f"Holter intervals: {len(holter_intervals)}")
    print(f"EDF mean interval: {np.mean(edf_intervals):.3f}s (HR: {60/np.mean(edf_intervals):.1f} bpm)")
    print(f"Holter mean interval: {np.mean(holter_intervals):.3f}s (HR: {60/np.mean(holter_intervals):.1f} bpm)")
    
    # Find best match using correlation
    print("\nFinding best match...")
    best_pos, scores, best_score = find_best_match(
        edf_intervals, 
        holter_intervals, 
        method='correlation',
        normalize=True
    )
    
    # The position in intervals corresponds to position+1 in peaks
    # (since first interval is between peak 0 and peak 1)
    peak_position = best_pos + 1
    time_position = edf_times[peak_position]
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Best match found at EDF peak index: {peak_position}")
    print(f"Holter recording starts at EDF time: {time_position:.3f} seconds")
    print(f"Correlation score: {best_score:.4f}")
    print(f"EDF peak sample at start: {edf_df.iloc[peak_position]['peak_sample']}")
    
    # Calculate end position
    end_peak_position = peak_position + len(holter_intervals)
    end_time_position = edf_times[end_peak_position]
    print(f"\nHolter recording ends at EDF peak index: {end_peak_position}")
    print(f"Holter recording ends at EDF time: {end_time_position:.3f} seconds")
    print(f"Duration: {end_time_position - time_position:.3f} seconds")
    
    # Show top 5 matches
    print("\n" + "-"*60)
    print("Top 5 matching positions:")
    print("-"*60)
    top_5_indices = np.argsort(scores)[-5:][::-1]
    for rank, idx in enumerate(top_5_indices, 1):
        peak_idx = idx + 1
        print(f"{rank}. Peak {peak_idx:4d} | Time {edf_times[peak_idx]:8.3f}s | Score {scores[idx]:.4f}")
    
    # Visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Similarity scores across all positions
    axes[0].plot(scores, linewidth=1)
    axes[0].axvline(best_pos, color='r', linestyle='--', linewidth=2, label=f'Best match (pos={best_pos})')
    axes[0].set_xlabel('Position in EDF (interval index)')
    axes[0].set_ylabel('Correlation Score')
    axes[0].set_title('Similarity Scores: Finding Holter Recording in EDF')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Comparison of matched intervals
    matched_edf = edf_intervals[best_pos:best_pos + len(holter_intervals)]
    x = np.arange(len(holter_intervals))
    axes[1].plot(x, holter_intervals * 1000, 'b.-', label='Holter', linewidth=2, markersize=4)
    axes[1].plot(x, matched_edf * 1000, 'r.-', label='Matched EDF', linewidth=2, markersize=4, alpha=0.7)
    axes[1].set_xlabel('Interval Index')
    axes[1].set_ylabel('RR Interval (ms)')
    axes[1].set_title(f'RR Interval Comparison at Best Match (Correlation: {best_score:.4f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Difference between matched intervals
    difference = (matched_edf - holter_intervals) * 1000  # in ms
    axes[2].plot(x, difference, 'g.-', linewidth=1, markersize=3)
    axes[2].axhline(0, color='k', linestyle='-', linewidth=0.5)
    axes[2].fill_between(x, 0, difference, alpha=0.3, color='g')
    axes[2].set_xlabel('Interval Index')
    axes[2].set_ylabel('Difference (ms)')
    axes[2].set_title(f'Difference Between Matched Intervals (Mean: {np.mean(difference):.2f} ms, Std: {np.std(difference):.2f} ms)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ecg_alignment_results.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to 'ecg_alignment_results.png'")
    plt.show()
    
    # Save detailed results
    results_df = pd.DataFrame({
        'holter_peak_index': range(len(holter_intervals)),
        'edf_peak_index': range(peak_position, peak_position + len(holter_intervals)),
        'holter_interval_ms': holter_intervals * 1000,
        'edf_interval_ms': matched_edf * 1000,
        'difference_ms': difference,
        'holter_time_sec': holter_times[:-1],  # Exclude last peak as it has no interval
        'edf_time_sec': edf_times[peak_position:peak_position + len(holter_intervals)]
    })
    
    results_df.to_csv('alignment_results.csv', index=False)
    print("Saved detailed results to 'alignment_results.csv'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Find where holter ECG recording starts within EDF recording by matching RR interval patterns.'
    )
    parser.add_argument(
        'edf_file',
        type=str,
        help='Path to EDF CSV file containing peak_sample and peak_time_sec columns'
    )
    parser.add_argument(
        'holter_file',
        type=str,
        help='Path to Holter CSV file containing peak_sample and peak_time_sec columns'
    )
    
    args = parser.parse_args()
    
    try:
        main(args.edf_file, args.holter_file)
    except FileNotFoundError as e:
        print(f"\nError: Could not find file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
