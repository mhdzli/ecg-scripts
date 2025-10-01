#!/usr/bin/env python3
"""
Enhanced ECG Beat Extractor CLI Tool
Extracts individual heartbeats from custom JSON ECG files using variable-length beat extraction
"""

import argparse
import json
import numpy as np
import os
import sys
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
from collections import defaultdict

class EnhancedECGBeatExtractor:
    def __init__(self, method='simple', extraction_method='variable_length', window_size=600, 
                 min_lead_agreement=1, tolerance_ms=100):
        self.method = method
        self.extraction_method = extraction_method  # 'fixed', 'adaptive', 'variable_length'
        self.window_size = window_size
        self.min_lead_agreement = min_lead_agreement
        self.tolerance_ms = tolerance_ms
        
    def load_ecg_json(self, filepath):
        """Load ECG data from custom JSON format"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Handle files with or without metadata
            metadata = data.get('metadata', None)
            leads = data.get('leads', data)  # If no 'leads' key, assume whole data is leads
            
            # Convert leads data to numpy arrays
            for lead_name in leads:
                if isinstance(leads[lead_name], list):
                    leads[lead_name] = np.array(leads[lead_name])
            
            return metadata, leads
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None
    
    def preprocess_ecg(self, signal, fs, lowcut=0.5, highcut=40):
        """Preprocess ECG signal with bandpass filtering"""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        if high >= 1.0:
            high = 0.99
        
        try:
            b, a = butter(4, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, signal)
            return filtered_signal
        except:
            print("Warning: Could not apply bandpass filter, using original signal")
            return signal
    
    def detect_r_peaks_simple(self, ecg_signal, fs):
        """Simple R-peak detection using scipy find_peaks"""
        min_distance = int(0.3 * fs)  # 300ms minimum distance
        prominence = np.std(ecg_signal) * 0.3
        height = np.mean(ecg_signal) + prominence
        
        peaks, properties = find_peaks(
            ecg_signal,
            height=height,
            distance=min_distance,
            prominence=prominence
        )
        
        return peaks
    
    def detect_r_peaks_pan_tompkins(self, ecg_signal, fs):
        """Simplified Pan-Tompkins algorithm"""
        # Bandpass filter (5-15 Hz)
        filtered = self.preprocess_ecg(ecg_signal, fs, 5, 15)
        
        # Derivative
        derivative = np.gradient(filtered)
        
        # Squaring
        squared = derivative ** 2
        
        # Moving window integration
        window_size = int(0.150 * fs)  # 150ms
        window = np.ones(window_size) / window_size
        integrated = np.convolve(squared, window, mode='same')
        
        # Peak detection
        min_distance = int(0.2 * fs)
        threshold = np.mean(integrated) + 2 * np.std(integrated)
        
        peaks, _ = find_peaks(
            integrated,
            height=threshold,
            distance=min_distance
        )
        
        # Refine peaks to original signal
        refined_peaks = []
        search_window = int(0.05 * fs)
        
        for peak in peaks:
            start = max(0, peak - search_window)
            end = min(len(ecg_signal), peak + search_window)
            local_max = np.argmax(ecg_signal[start:end]) + start
            refined_peaks.append(local_max)
        
        return np.array(refined_peaks)
    
    def detect_multi_lead_consensus(self, leads, fs, min_height_percentile=50):
        """
        Detect R-peaks using multi-lead consensus approach
        """
        detected_peaks = []
        tolerance_samples = int((self.tolerance_ms / 1000) * fs)
        
        # Step 1: QRS detection per lead
        for lead_name, signal in leads.items():
            # Skip if signal is too short or invalid
            if len(signal) < fs:  # Less than 1 second of data
                continue
                
            # Use absolute value for peak detection (handles inverted leads)
            abs_signal = np.abs(signal)
            
            # Adaptive threshold based on signal characteristics
            signal_std = np.std(abs_signal)
            signal_mean = np.mean(abs_signal)
            height_threshold = signal_mean + np.percentile(abs_signal, min_height_percentile) * 0.01
            
            try:
                if self.method == 'simple':
                    peaks = self.detect_r_peaks_simple(abs_signal, fs)
                elif self.method == 'pan_tompkins':
                    peaks = self.detect_r_peaks_pan_tompkins(signal, fs)
                else:
                    peaks = self.detect_r_peaks_simple(abs_signal, fs)
                
                # Refine peaks and add to detected peaks
                for peak in peaks:
                    if 0 <= peak < len(signal):
                        # Refine peak location in a small window
                        window_half = int(0.025 * fs)  # 25ms window
                        start = max(0, peak - window_half)
                        end = min(len(signal), peak + window_half)
                        local_segment = abs_signal[start:end]
                        
                        if len(local_segment) > 0 and np.max(local_segment) >= height_threshold:
                            refined_peak = start + np.argmax(local_segment)
                            detected_peaks.append((refined_peak, lead_name))
                            
            except Exception as e:
                print(f"Peak detection failed on lead {lead_name}: {e}")
                continue
        
        # Step 2: Cluster peaks within time tolerance
        if not detected_peaks:
            return []
            
        grouped_peaks = defaultdict(list)
        detected_peaks.sort(key=lambda x: x[0])
        
        for time_idx, lead_name in detected_peaks:
            matched = False
            for group_time in list(grouped_peaks.keys()):
                if abs(time_idx - group_time) <= tolerance_samples:
                    grouped_peaks[group_time].append((time_idx, lead_name))
                    matched = True
                    break
            if not matched:
                grouped_peaks[time_idx].append((time_idx, lead_name))
        
        # Step 3: Filter by consensus and return final peaks
        consensus_peaks = []
        for group_time, group in grouped_peaks.items():
            if len(group) >= self.min_lead_agreement:
                # Use median time as final peak location
                peak_times = [p[0] for p in group]
                consensus_peak = int(np.median(peak_times))
                consensus_peaks.append(consensus_peak)
        
        return sorted(consensus_peaks)
    
    def extract_beats_variable_length(self, leads, r_peaks, fs):
        """
        Extract beats with variable length based on RR intervals (like the working example)
        """
        if len(r_peaks) < 2:
            return []
            
        beats = []
        signal_length = len(next(iter(leads.values())))
        
        # Extract first beat: from start to midpoint between first and second peak
        first_end = r_peaks[0] + round((r_peaks[1] - r_peaks[0]) * 5.5 / 10)
        if first_end <= signal_length:
            beat_data = {}
            for lead_name, signal in leads.items():
                beat_data[lead_name] = signal[0:first_end].tolist()
            
            beats.append({
                'beat_number': 0,
                'peak_index': r_peaks[0],
                'leads': beat_data,
                'window_start': 0,
                'window_end': int(first_end),
                'window_size': first_end,
                'beat_type': 'first'
            })
        
        # Extract middle beats: from midpoint before to midpoint after
        for i in range(1, len(r_peaks) - 1):
            start = r_peaks[i] - round((r_peaks[i] - r_peaks[i-1]) * 0.45)
            end = r_peaks[i] + round((r_peaks[i+1] - r_peaks[i]) * 0.55)
            
            if start >= 0 and end <= signal_length and end > start:
                beat_data = {}
                for lead_name, signal in leads.items():
                    beat_data[lead_name] = signal[start:end].tolist()
                
                beats.append({
                    'beat_number': i,
                    'peak_index': r_peaks[i],
                    'leads': beat_data,
                    'window_start': int(start),
                    'window_end': int(end),
                    'window_size': int(end - start),
                    'beat_type': 'middle'
                })
        
        # Extract last beat: from midpoint before last peak to end
        if len(r_peaks) > 1:
            last_start = r_peaks[-1] - round((r_peaks[-1] - r_peaks[-2]) / 2)
            if last_start >= 0 and last_start < signal_length:
                beat_data = {}
                for lead_name, signal in leads.items():
                    beat_data[lead_name] = signal[last_start:].tolist()
                
                beats.append({
                    'beat_number': len(r_peaks) - 1,
                    'peak_index': r_peaks[-1],
                    'leads': beat_data,
                    'window_start': int(last_start),
                    'window_end': signal_length,
                    'window_size': signal_length - last_start,
                    'beat_type': 'last'
                })
        
        return beats
    
    def extract_beats_fixed_window(self, leads, r_peaks, window_size):
        """Extract beats with fixed window size"""
        beats = []
        half_window = window_size // 2
        signal_length = len(next(iter(leads.values())))
        
        for i, peak in enumerate(r_peaks):
            start = max(0, peak - half_window)
            end = min(signal_length, peak + half_window)
            
            if end - start == window_size:
                beat_data = {}
                for lead_name, signal in leads.items():
                    beat_data[lead_name] = signal[start:end].tolist()
                
                beats.append({
                    'beat_number': i,
                    'peak_index': peak,
                    'leads': beat_data,
                    'window_start': int(start),
                    'window_end': int(end),
                    'window_size': window_size,
                    'beat_type': 'fixed'
                })
        
        return beats
    
    def extract_beats_adaptive_window(self, leads, r_peaks, fs):
        """Extract beats with adaptive window sizes"""
        beats = []
        signal_length = len(next(iter(leads.values())))
        
        for i, peak in enumerate(r_peaks):
            # Calculate window size for this beat
            if i == 0 and len(r_peaks) > 1:
                # For first beat, use interval to next beat
                rr_interval = r_peaks[1] - r_peaks[0]
            elif i == len(r_peaks) - 1 and len(r_peaks) > 1:
                # For last beat, use interval from previous beat
                rr_interval = r_peaks[i] - r_peaks[i-1]
            elif len(r_peaks) > 2:
                # For middle beats, use average of surrounding intervals
                prev_interval = r_peaks[i] - r_peaks[i-1]
                next_interval = r_peaks[i+1] - r_peaks[i]
                rr_interval = (prev_interval + next_interval) / 2
            else:
                # Default window
                rr_interval = int(0.8 * fs)  # 800ms default
            
            # Calculate adaptive window (60% of RR interval)
            window_size = int(0.6 * rr_interval)
            window_size = max(int(0.2 * fs), min(int(1.2 * fs), window_size))
            half_window = window_size // 2
            
            start = max(0, peak - half_window)
            end = min(signal_length, peak + half_window)
            
            beat_data = {}
            for lead_name, signal in leads.items():
                beat_data[lead_name] = signal[start:end].tolist()
            
            beats.append({
                'beat_number': i,
                'peak_index': peak,
                'leads': beat_data,
                'window_start': int(start),
                'window_end': int(end),
                'window_size': len(beat_data[next(iter(beat_data.keys()))]),
                'rr_interval_samples': int(rr_interval),
                'beat_type': 'adaptive'
            })
        
        return beats
    
    def extract_beats_from_file(self, filepath, detection_lead='II'):
        """Extract beats from a single ECG JSON file"""
        metadata, leads = self.load_ecg_json(filepath)
        if leads is None:
            return None
        
        # Handle files without metadata - assume fs = 500
        if metadata is None or 'sampling_rate' not in metadata:
            fs = 500
            print(f"Warning: No metadata found in {filepath}, assuming fs=500Hz")
        else:
            fs = metadata['sampling_rate']
            if 'detection_lead' in metadata:
                detection_lead = metadata['detection_lead']
        
        # Find available lead for detection
        if detection_lead not in leads:
            available_leads = list(leads.keys())
            if available_leads:
                detection_lead = available_leads[0]
                print(f"Warning: Requested lead not found, using {detection_lead}")
            else:
                print("Error: No leads found in file")
                return None
        
        # Detect R-peaks
        if self.min_lead_agreement > 1:
            # Use multi-lead consensus
            r_peaks = self.detect_multi_lead_consensus(leads, fs)
        else:
            # Use single lead detection
            ecg_signal = leads[detection_lead]
            filtered_signal = self.preprocess_ecg(ecg_signal, fs)
            
            if self.method == 'simple':
                r_peaks = self.detect_r_peaks_simple(filtered_signal, fs)
            elif self.method == 'pan_tompkins':
                r_peaks = self.detect_r_peaks_pan_tompkins(ecg_signal, fs)
            else:
                raise ValueError(f"Unknown method: {self.method}")
        
        if len(r_peaks) == 0:
            print(f"Warning: No R-peaks detected in {filepath}")
            return []
        
        # Extract beats based on method
        if self.extraction_method == 'variable_length':
            beats = self.extract_beats_variable_length(leads, r_peaks, fs)
        elif self.extraction_method == 'adaptive':
            beats = self.extract_beats_adaptive_window(leads, r_peaks, fs)
        else:  # fixed
            beats = self.extract_beats_fixed_window(leads, r_peaks, self.window_size)
        
        # Add metadata to each beat
        for beat in beats:
            beat['metadata'] = {
                'original_file': str(filepath),
                'sampling_rate': fs,
                'detection_lead': detection_lead,
                'method': self.method,
                'extraction_method': self.extraction_method,
                'total_beats': len(beats),
                'total_r_peaks': len(r_peaks)
            }
        
        return beats
    
    def save_beats(self, beats, output_dir, base_filename):
        """Save extracted beats to separate JSON files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(base_filename).stem
        
        saved_files = []
        for beat in beats:
            beat_num = beat['beat_number']
            filename = f"{base_name}_{beat_num:04d}.json"
            filepath = output_path / filename
            
            # Convert numpy types to Python native types for JSON serialization
            beat_copy = self._convert_numpy_types(beat)
            
            with open(filepath, 'w') as f:
                json.dump(beat_copy, f, indent=2)
            
            saved_files.append(str(filepath))
        
        return saved_files
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced ECG Beat Extractor with variable-length beat extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract variable-length beats (recommended)
  python enhanced_ecg_extractor.py input.json -o output_dir --extraction variable_length
  
  # Use multi-lead consensus with variable-length extraction
  python enhanced_ecg_extractor.py input.json -o output_dir --extraction variable_length --consensus 3
  
  # Use adaptive window with Pan-Tompkins detection
  python enhanced_ecg_extractor.py input.json -o output_dir --extraction adaptive -m pan_tompkins
  
  # Process directory with fixed windows
  python enhanced_ecg_extractor.py input_dir/ -o output_dir --extraction fixed -w 500
        """
    )
    
    parser.add_argument('input', help='Input JSON file or directory')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory for extracted beats')
    parser.add_argument('-m', '--method', choices=['simple', 'pan_tompkins'],
                        default='simple', help='Peak detection method')
    parser.add_argument('--extraction', choices=['fixed', 'adaptive', 'variable_length'],
                        default='variable_length', help='Beat extraction method')
    parser.add_argument('-w', '--window-size', type=int, default=600,
                        help='Fixed window size in samples (for fixed extraction)')
    parser.add_argument('--consensus', type=int, default=1,
                        help='Minimum number of leads that must agree on peak location')
    parser.add_argument('--tolerance', type=int, default=100,
                        help='Time tolerance in milliseconds for peak consensus')
    parser.add_argument('-l', '--lead', default='II',
                        help='Lead to use for single-lead R-peak detection')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots for each processed file')
    parser.add_argument('--plot-beats', metavar='LEAD',
                        help='Plot all extracted heartbeats for specified lead')
    parser.add_argument('--analysis-lead', metavar='LEAD',
                        help='Plot analysis for specified lead')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = EnhancedECGBeatExtractor(
        method=args.method,
        extraction_method=args.extraction,
        window_size=args.window_size,
        min_lead_agreement=args.consensus,
        tolerance_ms=args.tolerance
    )
    
    input_path = Path(args.input)
    
    # Get list of files to process
    if input_path.is_file():
        files_to_process = [input_path]
    elif input_path.is_dir():
        files_to_process = list(input_path.glob('*.json'))
        if not files_to_process:
            print(f"No JSON files found in {input_path}")
            sys.exit(1)
    else:
        print(f"Input path {input_path} does not exist")
        sys.exit(1)
    
    total_beats = 0
    processed_files = 0
    
    for file_path in files_to_process:
        if args.verbose:
            print(f"Processing {file_path}...")
        
        try:
            beats = extractor.extract_beats_from_file(file_path, args.lead)
            
            if beats is None:
                print(f"Failed to process {file_path}")
                continue
            
            if len(beats) == 0:
                print(f"No beats found in {file_path}")
                continue
            
            # Save beats
            saved_files = extractor.save_beats(beats, args.output, file_path.name)
            
            total_beats += len(beats)
            processed_files += 1
            
            print(f"Extracted {len(beats)} beats from {file_path.name} (method: {args.extraction})")
            if args.verbose:
                beat_lengths = [beat['window_size'] for beat in beats]
                print(f"  Beat lengths: {min(beat_lengths)}-{max(beat_lengths)} samples")
                print(f"  Saved to: {len(saved_files)} files in {args.output}")
            
            # Generate plot if requested
            if args.plot:
                plot_file = Path(args.output) / f"{file_path.stem}_analysis.png"
                if args.analysis_lead:
                    a_lead = args.analysis_lead
                else:
                    a_lead = args.lead
                generate_analysis_plot(file_path, beats, plot_file, args.lead, a_lead)
            
            # Generate beats plot if requested
            if args.plot_beats:
                plot_file = Path(args.output) / f"{file_path.stem}_beats_{args.plot_beats}.png"
                generate_beats_plot(beats, plot_file, args.plot_beats, file_path.name)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    print(f"\nSummary:")
    print(f"Processed {processed_files} files")
    print(f"Extracted {total_beats} total beats")
    print(f"Extraction method: {args.extraction}")
    print(f"Output directory: {args.output}")

def generate_beats_plot(beats, output_file, lead, filename):
    """Generate plot showing all extracted heartbeats for a specific lead"""
    try:
        if not beats:
            print(f"  No beats to plot for {filename}")
            return
        
        # Check if requested lead exists in beats
        if lead not in beats[0]['leads']:
            available_leads = list(beats[0]['leads'].keys())
            print(f"  Lead {lead} not found. Available leads: {available_leads}")
            return
        
        # Calculate grid dimensions
        n_beats = len(beats)
        cols = min(5, n_beats)  # Max 5 columns
        rows = (n_beats + cols - 1) // cols  # Ceiling division
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each beat
        for i, beat in enumerate(beats):
            ax = axes[i] if rows > 1 or cols > 1 else axes[0]
            
            beat_signal = np.array(beat['leads'][lead])
            
            # For variable length beats, try to center around peak
            if beat.get('beat_type') in ['middle', 'adaptive']:
                # Find approximate peak location within the beat
                peak_in_beat = len(beat_signal) // 2
                x_axis = np.arange(len(beat_signal)) - peak_in_beat
            else:
                # For first/last beats or fixed windows
                x_axis = np.arange(len(beat_signal))
            
            ax.plot(x_axis, beat_signal, 'b-', linewidth=1.5)
            
            if beat.get('beat_type') in ['middle', 'adaptive']:
                ax.axvline(x=0, color='r', linestyle='--', alpha=0.7, linewidth=1)
            
            title = f"Beat {i+1}"
            if 'beat_type' in beat:
                title += f" ({beat['beat_type']})"
            
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Samples')
            ax.set_ylabel('Amplitude')
        
        # Hide unused subplots
        for i in range(n_beats, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'All Heartbeats - Lead {lead} ({filename})', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Beats plot saved: {output_file}")
        
    except Exception as e:
        print(f"  Failed to generate beats plot: {e}")

def generate_analysis_plot(input_file, beats, output_file, lead, a_lead):
    """Generate analysis plot showing original signal and detected beats"""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        leads = data.get('leads', data)
        fs = metadata.get('sampling_rate', 500)
        
        if lead not in leads:
            available_leads = list(leads.keys())
            if available_leads:
                lead = available_leads[0]
        
        if a_lead not in leads:
            available_leads = list(leads.keys())
            if available_leads:
                lead = available_leads[0]
        a_signal = np.array(leads[a_lead])
        signal = np.array(leads[lead])
        time = np.arange(len(signal)) / fs
        
        # Extract R-peak positions
        r_peaks = [beat['peak_index'] for beat in beats]
        
        plt.figure(figsize=(15, 10))
        
        # Plot original signal
        plt.subplot(3, 1, 1)
        plt.plot(time, signal, 'b-', alpha=0.7, label=f'ECG Lead {lead}')
        plt.plot(time[r_peaks], signal[r_peaks], 'ro', markersize=6, label='Detected R-peaks')
        plt.title(f'ECG Analysis: {input_file.name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot beat boundaries
        plt.subplot(3, 1, 2)
        plt.plot(time, a_signal, 'b-', alpha=0.7)
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, beat in enumerate(beats[:5]):  # Show first 5 beats
            start_time = beat['window_start'] / fs
            end_time = beat['window_end'] / fs
            color = colors[i % len(colors)]
            plt.axvspan(start_time, end_time, alpha=0.3, color=color, 
                       label=f"Beat {i+1} ({beat.get('beat_type', 'unknown')})")
        plt.title('Beat Segmentation (First 5 Beats)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot extracted beats overlaid
        plt.subplot(3, 1, 3)
        max_beats_to_show = min(10, len(beats))
        for i in range(max_beats_to_show):
            beat_signal = np.array(beats[i]['leads'][lead])
            # Normalize time axis for overlay
            if beats[i].get('beat_type') in ['middle', 'adaptive']:
                beat_time = np.arange(len(beat_signal))
            else:
                beat_time = np.arange(len(beat_signal))
            plt.plot(beat_time, beat_signal, alpha=0.7, label=f"Beat {i+1}")
        
        plt.title(f'Extracted Heartbeats Overlay (First {max_beats_to_show})')
        plt.xlabel('Samples from center/start')
        plt.ylabel('Amplitude')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Analysis plot saved: {output_file}")
        
    except Exception as e:
        print(f"  Failed to generate analysis plot: {e}")

if __name__ == '__main__':
    main()
