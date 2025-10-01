import pandas as pd
import numpy as np
from scipy import stats

def calculate_intervals(peak_times):
    """Calculate RR intervals from peak times"""
    return np.diff(peak_times)

def find_exact_match(edf_intervals, holter_intervals, tolerance=0.01):
    """
    Try to find exact match with tolerance (in seconds)
    Returns list of matching start positions
    """
    matches = []
    n_holter = len(holter_intervals)
    
    for i in range(len(edf_intervals) - n_holter + 1):
        edf_window = edf_intervals[i:i + n_holter]
        differences = np.abs(edf_window - holter_intervals)
        
        if np.all(differences < tolerance):
            matches.append(i)
    
    return matches

def calculate_match_score(edf_intervals, holter_intervals):
    """
    Calculate similarity score between two interval sequences
    Lower score = better match
    Returns mean absolute difference
    """
    return np.mean(np.abs(edf_intervals - holter_intervals))

def find_best_match(edf_intervals, holter_intervals):
    """
    Find best matching position using sliding window
    Returns position, score, and correlation
    """
    n_holter = len(holter_intervals)
    best_score = float('inf')
    best_position = -1
    best_correlation = -1
    
    scores = []
    
    for i in range(len(edf_intervals) - n_holter + 1):
        edf_window = edf_intervals[i:i + n_holter]
        
        # Calculate mean absolute error
        score = calculate_match_score(edf_window, holter_intervals)
        scores.append(score)
        
        # Calculate correlation
        if np.std(edf_window) > 0 and np.std(holter_intervals) > 0:
            correlation, _ = stats.pearsonr(edf_window, holter_intervals)
        else:
            correlation = 0
        
        if score < best_score:
            best_score = score
            best_position = i
            best_correlation = correlation
    
    return best_position, best_score, best_correlation, scores

def main():
    # Read CSV files
    print("Reading CSV files...")
    edf_data = pd.read_csv('EDF.csv')
    holter_data = pd.read_csv('holter.csv')
    
    print(f"EDF peaks: {len(edf_data)}")
    print(f"Holter peaks: {len(holter_data)}")
    
    # Calculate intervals
    print("\nCalculating intervals...")
    edf_intervals = calculate_intervals(edf_data['peak_time_sec'].values)
    holter_intervals = calculate_intervals(holter_data['peak_time_sec'].values)
    
    print(f"EDF intervals: {len(edf_intervals)}")
    print(f"Holter intervals: {len(holter_intervals)}")
    
    # Try exact match with increasing tolerances
    print("\n" + "="*60)
    print("TRYING EXACT MATCHES WITH TOLERANCE")
    print("="*60)
    
    tolerances = [0.001, 0.005, 0.01, 0.02, 0.05]
    exact_match_found = False
    
    for tol in tolerances:
        matches = find_exact_match(edf_intervals, holter_intervals, tolerance=tol)
        if matches:
            print(f"\n✓ Found {len(matches)} exact match(es) with tolerance {tol}s:")
            for match_idx in matches:
                edf_peak_idx = match_idx  # This is the peak index (0-based)
                edf_start_time = edf_data['peak_time_sec'].iloc[edf_peak_idx]
                holter_start_time = holter_data['peak_time_sec'].iloc[0]
                
                print(f"\n  Match at EDF peak index: {edf_peak_idx}")
                print(f"  EDF peak time: {edf_start_time:.3f} seconds")
                print(f"  Holter starts at: {holter_start_time:.3f} seconds")
                print(f"  Time offset: {edf_start_time - holter_start_time:.3f} seconds")
                
                # Show first few interval comparisons
                print(f"\n  First 5 interval comparisons:")
                for j in range(min(5, len(holter_intervals))):
                    edf_int = edf_intervals[match_idx + j]
                    holter_int = holter_intervals[j]
                    diff = abs(edf_int - holter_int)
                    print(f"    Interval {j}: EDF={edf_int:.4f}s, Holter={holter_int:.4f}s, Diff={diff:.4f}s")
            
            exact_match_found = True
            break
    
    if not exact_match_found:
        print("\n✗ No exact matches found with tested tolerances")
    
    # Find best match using correlation
    print("\n" + "="*60)
    print("FINDING BEST MATCH (PATTERN CORRELATION)")
    print("="*60)
    
    best_position, best_score, best_correlation, all_scores = find_best_match(
        edf_intervals, holter_intervals
    )
    
    edf_start_time = edf_data['peak_time_sec'].iloc[best_position]
    holter_start_time = holter_data['peak_time_sec'].iloc[0]
    
    print(f"\nBest match found at EDF peak index: {best_position}")
    print(f"EDF peak time: {edf_start_time:.3f} seconds")
    print(f"Holter starts at: {holter_start_time:.3f} seconds")
    print(f"Time offset: {edf_start_time - holter_start_time:.3f} seconds")
    print(f"Match score (MAE): {best_score:.4f} seconds")
    print(f"Correlation coefficient: {best_correlation:.4f}")
    
    # Show interval comparisons
    print(f"\nFirst 10 interval comparisons at best match:")
    for j in range(min(10, len(holter_intervals))):
        edf_int = edf_intervals[best_position + j]
        holter_int = holter_intervals[j]
        diff = abs(edf_int - holter_int)
        print(f"  Interval {j}: EDF={edf_int:.4f}s, Holter={holter_int:.4f}s, Diff={diff:.4f}s")
    
    # Show statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    
    all_scores = np.array(all_scores)
    print(f"\nMatch scores across all positions:")
    print(f"  Best (minimum): {np.min(all_scores):.4f}s")
    print(f"  Median: {np.median(all_scores):.4f}s")
    print(f"  Mean: {np.mean(all_scores):.4f}s")
    print(f"  Worst (maximum): {np.max(all_scores):.4f}s")
    
    # Find top 5 matches
    top_5_indices = np.argsort(all_scores)[:5]
    print(f"\nTop 5 best matching positions:")
    for rank, idx in enumerate(top_5_indices, 1):
        score = all_scores[idx]
        time = edf_data['peak_time_sec'].iloc[idx]
        print(f"  {rank}. Peak index {idx}, time {time:.3f}s, score {score:.4f}s")
    
    # Calculate quality of match
    print("\n" + "="*60)
    print("MATCH QUALITY ASSESSMENT")
    print("="*60)
    
    score_improvement = (np.median(all_scores) - best_score) / np.median(all_scores) * 100
    
    if best_correlation > 0.95 and best_score < 0.02:
        quality = "EXCELLENT"
    elif best_correlation > 0.9 and best_score < 0.05:
        quality = "GOOD"
    elif best_correlation > 0.8 and best_score < 0.1:
        quality = "MODERATE"
    else:
        quality = "POOR - Manual verification recommended"
    
    print(f"\nMatch Quality: {quality}")
    print(f"Score improvement over median: {score_improvement:.1f}%")
    
    if best_correlation > 0.9:
        print("\n✓ High correlation suggests reliable match")
    else:
        print("\n⚠ Low correlation - match may be unreliable")

if __name__ == "__main__":
    main()
