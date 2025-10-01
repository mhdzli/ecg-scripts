import pandas as pd
import numpy as np
import argparse
from scipy import stats

def load_peaks(filename):
    """Load peak data from CSV file."""
    df = pd.read_csv(filename)
    return df['peak_sample'].values, df['peak_time_sec'].values

def calculate_intervals_from_first(peak_times):
    """Calculate time intervals from the first peak to each subsequent peak."""
    if len(peak_times) == 0:
        return np.array([])
    return peak_times - peak_times[0]

def find_exact_matches(edf_intervals, holter_intervals, tolerance=0.01):
    """
    Find exact matches within tolerance.
    
    Args:
        edf_intervals: Array of intervals from candidate starting point in EDF
        holter_intervals: Array of intervals from first peak in Holter
        tolerance: Maximum allowed difference in seconds for exact match
    
    Returns:
        Boolean indicating if exact match found
    """
    if len(edf_intervals) < len(holter_intervals):
        return False
    
    # Compare the intervals
    differences = np.abs(edf_intervals[:len(holter_intervals)] - holter_intervals)
    
    # Check if all differences are within tolerance
    return np.all(differences <= tolerance)

def calculate_match_score(edf_intervals, holter_intervals):
    """
    Calculate match quality using multiple metrics.
    
    Returns:
        Dictionary with correlation, MSE, and max error
    """
    min_len = min(len(edf_intervals), len(holter_intervals))
    
    if min_len < 2:
        return {
            'correlation': 0,
            'mse': float('inf'),
            'max_error': float('inf'),
            'mean_error': float('inf')
        }
    
    edf_subset = edf_intervals[:min_len]
    holter_subset = holter_intervals[:min_len]
    
    # Calculate correlation
    if np.std(edf_subset) > 0 and np.std(holter_subset) > 0:
        correlation, _ = stats.pearsonr(edf_subset, holter_subset)
    else:
        correlation = 0
    
    # Calculate errors
    errors = np.abs(edf_subset - holter_subset)
    mse = np.mean(errors ** 2)
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    
    return {
        'correlation': correlation,
        'mse': mse,
        'max_error': max_error,
        'mean_error': mean_error
    }

def find_alignment(edf_samples, edf_times, holter_samples, holter_times, 
                   edf_sample_rate=250, holter_sample_rate=1000):
    """
    Find the alignment between EDF and Holter recordings.
    
    Args:
        edf_samples: Array of peak sample numbers in EDF
        edf_times: Array of peak times in seconds in EDF
        holter_samples: Array of peak sample numbers in Holter
        holter_times: Array of peak times in seconds in Holter
        edf_sample_rate: Sampling rate of EDF recording (Hz)
        holter_sample_rate: Sampling rate of Holter recording (Hz)
    
    Returns:
        Dictionary with alignment results
    """
    # Calculate intervals from first peak for Holter
    holter_intervals = calculate_intervals_from_first(holter_times)
    
    print(f"\nHolter recording:")
    print(f"  Number of peaks: {len(holter_times)}")
    print(f"  Duration from first to last peak: {holter_intervals[-1]:.2f} seconds")
    print(f"  First peak at: {holter_times[0]:.3f} seconds (sample {holter_samples[0]})")
    
    print(f"\nEDF recording:")
    print(f"  Number of peaks: {len(edf_times)}")
    print(f"  Total duration: {edf_times[-1] - edf_times[0]:.2f} seconds")
    
    # Search parameters
    num_holter_peaks = len(holter_times)
    max_search_idx = len(edf_times) - num_holter_peaks + 1
    
    if max_search_idx <= 0:
        print("\nError: EDF has fewer peaks than Holter recording!")
        return None
    
    print(f"\nSearching through {max_search_idx} possible starting positions...")
    
    # Storage for results
    exact_matches = []
    best_match = None
    best_score = float('inf')
    
    # Try different tolerances for exact matching
    tolerances = [0.005, 0.01, 0.02, 0.05]
    
    # Search through all possible starting positions
    for start_idx in range(max_search_idx):
        # Get EDF intervals starting from this position
        edf_candidate_times = edf_times[start_idx:start_idx + num_holter_peaks]
        edf_intervals = calculate_intervals_from_first(edf_candidate_times)
        
        # Calculate match score
        score = calculate_match_score(edf_intervals, holter_intervals)
        
        # Check for exact matches at different tolerances
        for tol in tolerances:
            if find_exact_matches(edf_intervals, holter_intervals, tolerance=tol):
                exact_matches.append({
                    'start_idx': start_idx,
                    'start_sample': edf_samples[start_idx],
                    'start_time': edf_times[start_idx],
                    'tolerance': tol,
                    'score': score
                })
                break
        
        # Track best match based on MSE
        if score['mse'] < best_score:
            best_score = score['mse']
            best_match = {
                'start_idx': start_idx,
                'start_sample': edf_samples[start_idx],
                'start_time': edf_times[start_idx],
                'score': score,
                'edf_intervals': edf_intervals,
                'holter_intervals': holter_intervals
            }
    
    # Report results
    print("\n" + "="*70)
    if exact_matches:
        print(f"FOUND {len(exact_matches)} EXACT MATCH(ES)!")
        print("="*70)
        
        # Report the best exact match (smallest tolerance)
        best_exact = min(exact_matches, key=lambda x: x['tolerance'])
        print(f"\nBest exact match (tolerance: ±{best_exact['tolerance']*1000:.1f} ms):")
        print(f"  EDF Peak Index: {best_exact['start_idx']}")
        print(f"  EDF Peak Sample: {best_exact['start_sample']}")
        print(f"  EDF Peak Time: {best_exact['start_time']:.3f} seconds")
        print(f"  Correlation: {best_exact['score']['correlation']:.6f}")
        print(f"  Mean Squared Error: {best_exact['score']['mse']:.6f} s²")
        print(f"  Mean Error: {best_exact['score']['mean_error']*1000:.2f} ms")
        print(f"  Max Error: {best_exact['score']['max_error']*1000:.2f} ms")
        
        # Convert to Holter sample space (accounting for different sample rates)
        sample_rate_ratio = holter_sample_rate / edf_sample_rate
        holter_equivalent_sample = int(best_exact['start_sample'] * sample_rate_ratio)
        print(f"\n  Equivalent Holter sample number: {holter_equivalent_sample}")
        print(f"  (EDF sample {best_exact['start_sample']} × {sample_rate_ratio})")
        
        if len(exact_matches) > 1:
            print(f"\n  Note: {len(exact_matches)} exact matches found at different tolerances")
        
        return best_exact
    else:
        print("NO EXACT MATCHES FOUND - Showing best approximate match:")
        print("="*70)
        print(f"\nBest approximate match:")
        print(f"  EDF Peak Index: {best_match['start_idx']}")
        print(f"  EDF Peak Sample: {best_match['start_sample']}")
        print(f"  EDF Peak Time: {best_match['start_time']:.3f} seconds")
        print(f"  Correlation: {best_match['score']['correlation']:.6f}")
        print(f"  Mean Squared Error: {best_match['score']['mse']:.6f} s²")
        print(f"  Mean Error: {best_match['score']['mean_error']*1000:.2f} ms")
        print(f"  Max Error: {best_match['score']['max_error']*1000:.2f} ms")
        
        # Convert to Holter sample space
        sample_rate_ratio = holter_sample_rate / edf_sample_rate
        holter_equivalent_sample = int(best_match['start_sample'] * sample_rate_ratio)
        print(f"\n  Equivalent Holter sample number: {holter_equivalent_sample}")
        print(f"  (EDF sample {best_match['start_sample']} × {sample_rate_ratio})")
        
        # Show first few interval comparisons
        print("\n  First 10 interval comparisons (seconds):")
        print("  Index | Holter Interval | EDF Interval | Difference")
        print("  " + "-"*60)
        for i in range(min(10, len(best_match['holter_intervals']))):
            diff = best_match['edf_intervals'][i] - best_match['holter_intervals'][i]
            print(f"  {i:5d} | {best_match['holter_intervals'][i]:14.6f} | "
                  f"{best_match['edf_intervals'][i]:12.6f} | {diff:+10.6f}")
        
        return best_match

def main():
    parser = argparse.ArgumentParser(
        description='Find alignment between EDF and Holter ECG recordings by comparing peak intervals.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py -e /path/to/EDF.csv -H /path/to/holter.csv
  python script.py -e data/EDF.csv -H data/holter.csv --edf-rate 250 --holter-rate 1000
        """
    )
    parser.add_argument('-e', '--edf', required=True, 
                       help='Path to EDF peaks CSV file')
    parser.add_argument('-H', '--holter', required=True, 
                       help='Path to Holter peaks CSV file')
    parser.add_argument('--edf-rate', type=int, default=250, 
                       help='EDF sampling rate in Hz (default: 250)')
    parser.add_argument('--holter-rate', type=int, default=1000,
                       help='Holter sampling rate in Hz (default: 1000)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ECG Peak Alignment Finder")
    print("="*70)
    
    # Load data
    print(f"\nLoading data...")
    print(f"  EDF file: {args.edf}")
    print(f"  Holter file: {args.holter}")
    
    edf_samples, edf_times = load_peaks(args.edf)
    holter_samples, holter_times = load_peaks(args.holter)
    
    # Find alignment
    result = find_alignment(
        edf_samples, edf_times, 
        holter_samples, holter_times,
        edf_sample_rate=args.edf_rate,
        holter_sample_rate=args.holter_rate
    )
    
    if result:
        print("\n" + "="*70)
        print("Analysis complete!")
        print("="*70)

if __name__ == "__main__":
    main()
