import json
import numpy as np
import os
import argparse
from scipy import signal
import matplotlib.pyplot as plt
import glob

class PanTompkinsDetector:
    """
    Pan-Tompkins QRS detection algorithm implementation.
    """
    
    def __init__(self, sampling_rate=500):
        self.fs = sampling_rate
        self.setup_filters()
        
    def setup_filters(self):
        """Setup bandpass and derivative filters for Pan-Tompkins algorithm."""
        # Bandpass filter: 5-15 Hz
        # Low-pass filter coefficients (cutoff ~15 Hz)
        self.lp_b = np.array([1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]) / 32
        self.lp_a = np.array([1, -2, 1])
        
        # High-pass filter coefficients (cutoff ~5 Hz)
        self.hp_b = np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, -32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) / 32
        self.hp_a = np.array([1, -1])
        
        # Derivative filter coefficients
        self.deriv_b = np.array([1, 2, 0, -2, -1]) / 8
        
    def bandpass_filter(self, signal_data):
        """Apply bandpass filter (5-15 Hz)."""
        # Apply low-pass filter
        lp_filtered = signal.lfilter(self.lp_b, self.lp_a, signal_data)
        # Apply high-pass filter
        bp_filtered = signal.lfilter(self.hp_b, self.hp_a, lp_filtered)
        return bp_filtered
    
    def derivative_filter(self, signal_data):
        """Apply derivative filter."""
        return signal.lfilter(self.deriv_b, [1], signal_data)
    
    def squaring(self, signal_data):
        """Square the signal."""
        return signal_data ** 2
    
    def moving_window_integration(self, signal_data, window_size=None):
        """Apply moving window integration."""
        if window_size is None:
            # Default window size: 150ms worth of samples
            window_size = int(0.15 * self.fs)
        
        # Ensure window_size is odd
        if window_size % 2 == 0:
            window_size += 1
            
        # Moving average filter
        return signal.convolve(signal_data, np.ones(window_size)/window_size, mode='same')
    
    def find_peaks_adaptive(self, signal_data, min_distance=None):
        """
        Find peaks using adaptive thresholding as in Pan-Tompkins algorithm.
        """
        if min_distance is None:
            # Minimum distance between peaks (200ms = typical minimum RR interval)
            min_distance = int(0.2 * self.fs)
        
        # Initialize thresholds
        spki = 0  # Signal peak
        npki = 0  # Noise peak
        threshold1 = 0
        threshold2 = 0
        
        # Learning phase - find initial peaks
        peaks = []
        rr_intervals = []
        
        # Simple peak detection for initialization
        initial_peaks, _ = signal.find_peaks(signal_data, 
                                           height=np.max(signal_data) * 0.3,
                                           distance=min_distance)
        
        if len(initial_peaks) < 2:
            # If no peaks found, use a lower threshold
            initial_peaks, _ = signal.find_peaks(signal_data, 
                                               height=np.max(signal_data) * 0.1,
                                               distance=min_distance)
        
        # Initialize thresholds based on initial peaks
        if len(initial_peaks) > 0:
            peak_heights = signal_data[initial_peaks]
            spki = np.mean(peak_heights)
            npki = np.mean(peak_heights) * 0.5
            threshold1 = npki + 0.25 * (spki - npki)
            threshold2 = 0.5 * threshold1
        
        # Adaptive peak detection
        for i, peak_idx in enumerate(initial_peaks):
            peak_height = signal_data[peak_idx]
            
            # Check if peak exceeds threshold
            if peak_height > threshold1:
                peaks.append(peak_idx)
                
                # Update signal peak
                spki = 0.125 * peak_height + 0.875 * spki
                
                # Calculate RR interval
                if len(peaks) > 1:
                    rr_interval = peaks[-1] - peaks[-2]
                    rr_intervals.append(rr_interval)
                
            else:
                # Update noise peak
                npki = 0.125 * peak_height + 0.875 * npki
            
            # Update thresholds
            threshold1 = npki + 0.25 * (spki - npki)
            threshold2 = 0.5 * threshold1
        
        return np.array(peaks), np.array(rr_intervals)
    
    def detect_qrs(self, ecg_signal):
        """
        Complete Pan-Tompkins QRS detection pipeline.
        """
        # Step 1: Bandpass filter (5-15 Hz)
        filtered = self.bandpass_filter(ecg_signal)
        
        # Step 2: Derivative filter
        derivative = self.derivative_filter(filtered)
        
        # Step 3: Squaring
        squared = self.squaring(derivative)
        
        # Step 4: Moving window integration
        integrated = self.moving_window_integration(squared)
        
        # Step 5: Adaptive peak detection
        peaks, rr_intervals = self.find_peaks_adaptive(integrated)
        
        return {
            'peaks': peaks,
            'rr_intervals': rr_intervals,
            'filtered_signal': filtered,
            'derivative': derivative,
            'squared': squared,
            'integrated': integrated
        }

def extract_beats_from_json(json_file, output_folder, 
                           lead_name='II', 
                           beat_window_ms=600, 
                           min_rr_ms=300, 
                           max_rr_ms=2000,
                           plot_detection=False):
    """
    Extract individual beats from ECG JSON file using Pan-Tompkins algorithm.
    
    Args:
        json_file (str): Path to ECG JSON file
        output_folder (str): Output folder for beat files
        lead_name (str): ECG lead to use for beat detection
        beat_window_ms (int): Window size around each beat (milliseconds)
        min_rr_ms (int): Minimum RR interval (milliseconds)
        max_rr_ms (int): Maximum RR interval (milliseconds)
        plot_detection (bool): Whether to plot detection results
    """
    
    print(f"Processing {json_file}...")
    
    # Load ECG data
    with open(json_file, 'r') as f:
        ecg_data = json.load(f)
    
    # Extract metadata
    if 'metadata' in ecg_data:
        metadata = ecg_data['metadata']
        leads = ecg_data['leads']
        sampling_rate = metadata['sampling_rate']
    else:
        # Handle older format
        leads = {k: v for k, v in ecg_data.items() if k != 'sampling_rate'}
        sampling_rate = ecg_data['sampling_rate']
        metadata = {'sampling_rate': sampling_rate}
    
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Available leads: {', '.join(leads.keys())}")
    
    # Check if requested lead exists
    if lead_name not in leads:
        print(f"Warning: Lead {lead_name} not found. Using first available lead.")
        lead_name = list(leads.keys())[0]
    
    # Get ECG signal for beat detection
    ecg_signal = np.array(leads[lead_name])
    print(f"Using lead {lead_name} for beat detection")
    print(f"Signal length: {len(ecg_signal)} samples ({len(ecg_signal)/sampling_rate:.2f} seconds)")
    
    # Initialize Pan-Tompkins detector
    detector = PanTompkinsDetector(sampling_rate)
    
    # Detect QRS complexes
    print("Running Pan-Tompkins QRS detection...")
    detection_results = detector.detect_qrs(ecg_signal)
    peaks = detection_results['peaks']
    rr_intervals = detection_results['rr_intervals']
    
    print(f"Found {len(peaks)} potential QRS complexes")
    
    # Filter peaks based on RR interval constraints
    min_rr_samples = int(min_rr_ms * sampling_rate / 1000)
    max_rr_samples = int(max_rr_ms * sampling_rate / 1000)
    
    valid_peaks = []
    for i, peak in enumerate(peaks):
        if i == 0:
            valid_peaks.append(peak)
        else:
            rr_interval = peak - peaks[i-1]
            if min_rr_samples <= rr_interval <= max_rr_samples:
                valid_peaks.append(peak)
    
    valid_peaks = np.array(valid_peaks)
    print(f"After RR interval filtering: {len(valid_peaks)} valid beats")
    
    if len(valid_peaks) == 0:
        print("No valid beats found!")
        return
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Calculate beat window in samples
    beat_window_samples = int(beat_window_ms * sampling_rate / 1000)
    half_window = beat_window_samples // 2
    
    # Extract individual beats
    beats_data = []
    valid_beat_count = 0
    prefix = os.path.splitext(json_file)[0]
    for i, peak_idx in enumerate(valid_peaks):
        # Define beat window
        start_idx = max(0, peak_idx - half_window)
        end_idx = min(len(ecg_signal), peak_idx + half_window)
        
        # Skip if window is too small
        if end_idx - start_idx < beat_window_samples * 0.8:
            continue
        
        # Extract beat data for all leads
        beat_leads = {}
        for lead, lead_data in leads.items():
            beat_leads[lead] = lead_data[start_idx:end_idx]
        
        # Beat metadata
        beat_metadata = {
            'beat_number': valid_beat_count,
            'peak_index': int(peak_idx),
            'start_index': int(start_idx),
            'end_index': int(end_idx),
            'peak_time_seconds': float(peak_idx / sampling_rate),
            'beat_duration_ms': len(beat_leads[lead_name]) * 1000 / sampling_rate,
            'sampling_rate': sampling_rate,
            'detection_lead': lead_name,
            'source_file': json_file
        }
        
        # Add RR interval if available
        if i > 0:
            rr_interval_samples = peak_idx - valid_peaks[i-1]
            beat_metadata['rr_interval_ms'] = float(rr_interval_samples * 1000 / sampling_rate)
            beat_metadata['heart_rate_bpm'] = 60000 / beat_metadata['rr_interval_ms']
        
        # Create beat data structure
        beat_data = {
            'metadata': beat_metadata,
            'leads': beat_leads
        }
        
        beats_data.append(beat_data)
        
        # Save individual beat file
        beat_filename = f"{output_folder}/{prefix}_beat_{valid_beat_count:04d}.json"
        with open(beat_filename, 'w') as f:
            json.dump(beat_data, f, indent=2)
        
        valid_beat_count += 1
    
    print(f"Extracted {valid_beat_count} beats to '{output_folder}/'")
    
    # Save summary file
    summary_data = {
        'source_file': json_file,
        'detection_parameters': {
            'detection_lead': lead_name,
            'beat_window_ms': beat_window_ms,
            'min_rr_ms': min_rr_ms,
            'max_rr_ms': max_rr_ms,
            'sampling_rate': sampling_rate
        },
        'results': {
            'total_beats_detected': len(peaks),
            'valid_beats_extracted': valid_beat_count,
            'average_rr_interval_ms': float(np.mean(rr_intervals) * 1000 / sampling_rate) if len(rr_intervals) > 0 else None,
            'average_heart_rate_bpm': 60000 / (np.mean(rr_intervals) * 1000 / sampling_rate) if len(rr_intervals) > 0 else None
        },
        'beat_files': [f"beat_{i:04d}.json" for i in range(valid_beat_count)]
    }
    
    summary_filename = f"{output_folder}/{prefix}_beats_summary.json"
    with open(summary_filename, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Plot detection results if requested
    if plot_detection:
        plot_detection_results(ecg_signal, detection_results, valid_peaks, 
                             sampling_rate, output_folder, lead_name)
    
    return beats_data

def plot_detection_results(ecg_signal, detection_results, valid_peaks, 
                         sampling_rate, output_folder, lead_name):
    """Plot Pan-Tompkins detection results."""
    
    time_axis = np.arange(len(ecg_signal)) / sampling_rate
    
    fig, axes = plt.subplots(5, 1, figsize=(15, 12))
    fig.suptitle(f'Pan-Tompkins QRS Detection Results - Lead {lead_name}', fontsize=16)
    
    # Original signal
    axes[0].plot(time_axis, ecg_signal, 'b-', linewidth=0.8)
    axes[0].plot(time_axis[valid_peaks], ecg_signal[valid_peaks], 'ro', markersize=4)
    axes[0].set_title('Original ECG Signal with Detected Beats')
    axes[0].set_ylabel('Amplitude (mV)')
    axes[0].grid(True, alpha=0.3)
    
    # Filtered signal
    axes[1].plot(time_axis, detection_results['filtered_signal'], 'g-', linewidth=0.8)
    axes[1].set_title('Bandpass Filtered (5-15 Hz)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    # Derivative
    axes[2].plot(time_axis, detection_results['derivative'], 'orange', linewidth=0.8)
    axes[2].set_title('Derivative Filter Output')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    
    # Squared
    axes[3].plot(time_axis, detection_results['squared'], 'purple', linewidth=0.8)
    axes[3].set_title('Squared Signal')
    axes[3].set_ylabel('Amplitude')
    axes[3].grid(True, alpha=0.3)
    
    # Integrated with peaks
    axes[4].plot(time_axis, detection_results['integrated'], 'r-', linewidth=0.8)
    axes[4].plot(time_axis[valid_peaks], detection_results['integrated'][valid_peaks], 'ko', markersize=4)
    axes[4].set_title('Moving Window Integration with Detected Peaks')
    axes[4].set_ylabel('Amplitude')
    axes[4].set_xlabel('Time (seconds)')
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{output_folder}/detection_results.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detection plot saved to {plot_filename}")

def main():
    parser = argparse.ArgumentParser(description="Extract ECG beats using Pan-Tompkins algorithm")
    
    # Change input_file to input_path to handle both files and directories
    parser.add_argument("input_path", help="Input ECG JSON file or directory containing JSON files")
    parser.add_argument("-o", "--output", default="extracted_beats", 
                       help="Output folder for beat files (default: extracted_beats)")
    parser.add_argument("-l", "--lead", default="II", 
                       help="ECG lead for beat detection (default: II)")
    parser.add_argument("-w", "--window", type=int, default=600,
                       help="Beat window size in milliseconds (default: 600)")
    parser.add_argument("--min-rr", type=int, default=300,
                       help="Minimum RR interval in milliseconds (default: 300)")
    parser.add_argument("--max-rr", type=int, default=2000,
                       help="Maximum RR interval in milliseconds (default: 2000)")
    parser.add_argument("--plot", action="store_true",
                       help="Generate detection result plots")
    
    args = parser.parse_args()
    
    # Determine if input is a file or directory
    if os.path.isfile(args.input_path):
        # Single file processing
        json_files = [args.input_path]
        print("=== ECG Beat Extractor with Pan-Tompkins Algorithm ===")
        print(f"Processing single file: {args.input_path}")
    elif os.path.isdir(args.input_path):
        # Directory processing - find all JSON files
        json_pattern = os.path.join(args.input_path, "*.json")
        json_files = glob.glob(json_pattern)
        
        if not json_files:
            print(f"Error: No JSON files found in directory '{args.input_path}'!")
            return
        
        print("=== ECG Beat Extractor with Pan-Tompkins Algorithm (Batch Mode) ===")
        print(f"Processing directory: {args.input_path}")
        print(f"Found {len(json_files)} JSON files:")
        for file in json_files:
            print(f"  - {os.path.basename(file)}")
    else:
        print(f"Error: Input path '{args.input_path}' is neither a file nor a directory!")
        return
    
    print(f"Output folder: {args.output}")
    print(f"Detection lead: {args.lead}")
    print(f"Beat window: {args.window} ms")
    print(f"RR interval range: {args.min_rr}-{args.max_rr} ms")
    print()
    
    # Process each JSON file
    total_beats = 0
    successful_files = 0
    
    for i, json_file in enumerate(json_files, 1):
        try:
            print(f"=== Processing file {i}/{len(json_files)}: {os.path.basename(json_file)} ===")
            
            # Create individual output folder for each file
            base_name = os.path.splitext(os.path.basename(json_file))[0]
            file_output_folder = os.path.join(args.output, f"{base_name}_beats")
            
            # Extract beats
            beats = extract_beats_from_json(
                json_file, 
                file_output_folder,
                lead_name=args.lead,
                beat_window_ms=args.window,
                min_rr_ms=args.min_rr,
                max_rr_ms=args.max_rr,
                plot_detection=args.plot
            )
            
            if beats:
                total_beats += len(beats)
                successful_files += 1
                print(f"✓ Successfully extracted {len(beats)} beats from {os.path.basename(json_file)}")
            else:
                print(f"⚠ No beats extracted from {os.path.basename(json_file)}")
            
            print()  # Add spacing between files
            
        except Exception as e:
            print(f"✗ ERROR processing {os.path.basename(json_file)}: {str(e)}")
            print()
            continue
    
    # Print summary
    print("=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {len(json_files)}")
    print(f"Successful extractions: {successful_files}")
    print(f"Failed extractions: {len(json_files) - successful_files}")
    print(f"Total beats extracted: {total_beats}")
    if successful_files > 0:
        print(f"Average beats per file: {total_beats / successful_files:.1f}")
    print(f"Output location: {args.output}/")
    print("\nBeat extraction complete!")


if __name__ == "__main__":
    main()
