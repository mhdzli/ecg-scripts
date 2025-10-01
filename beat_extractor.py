import json
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple, Optional

class PanTompkinsDetector:
    """
    Pan-Tompkins QRS detection algorithm implementation
    """
    def __init__(self, sampling_rate: int = 500):
        self.sampling_rate = sampling_rate
        self.window_size = int(0.15 * sampling_rate)  # 150ms integration window
        
    def bandpass_filter(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter (5-15 Hz)"""
        nyquist = self.sampling_rate / 2
        low = 5 / nyquist
        high = 15 / nyquist
        b, a = butter(2, [low, high], btype='band')
        return filtfilt(b, a, signal_data)
    
    def derivative_filter(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply derivative filter to emphasize QRS slope"""
        # 5-point derivative: y(n) = (1/8T)[-x(n-2) - 2x(n-1) + 2x(n+1) + x(n+2)]
        derivative = np.zeros_like(signal_data)
        for i in range(2, len(signal_data) - 2):
            derivative[i] = (-signal_data[i-2] - 2*signal_data[i-1] + 
                           2*signal_data[i+1] + signal_data[i+2]) / 8
        return derivative
    
    def squaring(self, signal_data: np.ndarray) -> np.ndarray:
        """Square the signal to make all peaks positive and emphasize higher frequencies"""
        return signal_data ** 2
    
    def moving_window_integration(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply moving window integration"""
        integrated = np.zeros_like(signal_data)
        for i in range(len(signal_data)):
            start = max(0, i - self.window_size + 1)
            integrated[i] = np.sum(signal_data[start:i+1]) / self.window_size
        return integrated
    
    def adaptive_threshold(self, integrated_signal: np.ndarray, 
                          peaks: np.ndarray) -> Tuple[float, float]:
        """Calculate adaptive thresholds"""
        if len(peaks) == 0:
            return np.max(integrated_signal) * 0.25, np.max(integrated_signal) * 0.125
        
        # Get peak values
        peak_values = integrated_signal[peaks]
        
        # Signal and noise peak estimates
        spki = np.mean(peak_values[-8:]) if len(peak_values) >= 8 else np.mean(peak_values)
        npki = np.mean(np.sort(peak_values)[:len(peak_values)//2]) if len(peak_values) > 1 else spki * 0.5
        
        # Thresholds
        threshold1 = npki + 0.25 * (spki - npki)
        threshold2 = 0.5 * threshold1
        
        return threshold1, threshold2
    
    def detect_qrs(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Main Pan-Tompkins QRS detection function
        Returns indices of detected QRS complexes
        """
        # Step 1: Bandpass filter
        filtered = self.bandpass_filter(ecg_signal)
        
        # Step 2: Derivative
        derivative = self.derivative_filter(filtered)
        
        # Step 3: Squaring
        squared = self.squaring(derivative)
        
        # Step 4: Moving window integration
        integrated = self.moving_window_integration(squared)
        
        # Step 5: Find peaks with adaptive thresholding
        # Initial peak detection
        min_distance = int(0.3 * self.sampling_rate)  # Minimum 300ms between peaks
        initial_peaks, _ = find_peaks(integrated, distance=min_distance)
        
        if len(initial_peaks) == 0:
            return np.array([])
        
        # Adaptive thresholding
        threshold1, threshold2 = self.adaptive_threshold(integrated, initial_peaks)
        
        # Final peak detection with thresholds
        final_peaks = []
        for peak in initial_peaks:
            if integrated[peak] > threshold1:
                final_peaks.append(peak)
        
        # Search back for missed beats
        for i in range(1, len(final_peaks)):
            interval = final_peaks[i] - final_peaks[i-1]
            if interval > 1.66 * np.mean(np.diff(final_peaks)):  # If interval is too long
                # Look for peaks above threshold2 in the interval
                start_search = final_peaks[i-1] + int(0.2 * self.sampling_rate)
                end_search = final_peaks[i] - int(0.2 * self.sampling_rate)
                
                if start_search < end_search:
                    search_region = integrated[start_search:end_search]
                    local_peaks, _ = find_peaks(search_region)
                    
                    for local_peak in local_peaks:
                        global_peak = start_search + local_peak
                        if integrated[global_peak] > threshold2:
                            final_peaks.append(global_peak)
        
        return np.array(sorted(final_peaks))

class ECGBeatExtractor:
    """
    Extract individual heartbeat segments from ECG data
    """
    def __init__(self, sampling_rate: int = 500):
        self.sampling_rate = sampling_rate
        self.detector = PanTompkinsDetector(sampling_rate)
        
        # Beat extraction parameters
        self.pre_r_samples = int(0.2 * sampling_rate)   # 200ms before R-peak
        self.post_r_samples = int(0.4 * sampling_rate)  # 400ms after R-peak
        self.beat_length = self.pre_r_samples + self.post_r_samples
        
    def load_ecg_json(self, json_file: str) -> Dict:
        """Load ECG data from JSON file"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    
    def extract_beats_from_lead(self, lead_data: List[float], lead_name: str) -> Dict:
        """Extract beats from a single ECG lead"""
        ecg_signal = np.array(lead_data)
        
        # Detect QRS complexes
        r_peaks = self.detector.detect_qrs(ecg_signal)
        
        beats = []
        valid_beats = 0
        
        for i, r_peak in enumerate(r_peaks):
            # Check if we have enough samples before and after R-peak
            start_idx = r_peak - self.pre_r_samples
            end_idx = r_peak + self.post_r_samples
            
            if start_idx >= 0 and end_idx < len(ecg_signal):
                beat_segment = ecg_signal[start_idx:end_idx]
                
                beats.append({
                    'beat_index': valid_beats,
                    'r_peak_global_index': int(r_peak),
                    'r_peak_time_seconds': float(r_peak / self.sampling_rate),
                    'beat_segment': beat_segment.tolist(),
                    'beat_length_samples': len(beat_segment),
                    'pre_r_samples': self.pre_r_samples,
                    'post_r_samples': self.post_r_samples
                })
                valid_beats += 1
        
        # Calculate RR intervals
        rr_intervals = []
        if len(r_peaks) > 1:
            rr_intervals = np.diff(r_peaks) / self.sampling_rate  # Convert to seconds
        
        return {
            'lead_name': lead_name,
            'total_r_peaks_detected': len(r_peaks),
            'valid_beats_extracted': valid_beats,
            'r_peak_indices': r_peaks.tolist(),
            'mean_rr_interval_seconds': float(np.mean(rr_intervals)) if len(rr_intervals) > 0 else None,
            'heart_rate_bpm': float(60 / np.mean(rr_intervals)) if len(rr_intervals) > 0 else None,
            'beats': beats
        }
    
    def extract_beats_from_json(self, json_file: str, output_file: Optional[str] = None,
                              leads_to_process: Optional[List[str]] = None) -> Dict:
        """
        Extract beats from all leads in ECG JSON file
        """
        print(f"Loading ECG data from {json_file}...")
        ecg_data = self.load_ecg_json(json_file)
        
        # Get available leads
        available_leads = list(ecg_data['leads'].keys())
        
        # Determine which leads to process
        if leads_to_process is None:
            leads_to_process = available_leads
        else:
            leads_to_process = [lead for lead in leads_to_process if lead in available_leads]
        
        print(f"Processing leads: {leads_to_process}")
        
        # Extract beats from each lead
        extracted_data = {
            'metadata': {
                'source_file': json_file,
                'sampling_rate': ecg_data['metadata']['sampling_rate'],
                'total_duration_seconds': ecg_data['metadata']['duration_seconds'],
                'extraction_parameters': {
                    'pre_r_samples': self.pre_r_samples,
                    'post_r_samples': self.post_r_samples,
                    'beat_length_samples': self.beat_length,
                    'pre_r_duration_ms': self.pre_r_samples / self.sampling_rate * 1000,
                    'post_r_duration_ms': self.post_r_samples / self.sampling_rate * 1000,
                    'beat_duration_ms': self.beat_length / self.sampling_rate * 1000
                },
                'leads_processed': leads_to_process,
                'algorithm': 'Pan-Tompkins QRS Detection'
            },
            'leads': {}
        }
        
        for lead_name in leads_to_process:
            print(f"Processing lead {lead_name}...")
            lead_data = ecg_data['leads'][lead_name]
            
            lead_beats = self.extract_beats_from_lead(lead_data, lead_name)
            extracted_data['leads'][lead_name] = lead_beats
            
            print(f"  - Detected {lead_beats['total_r_peaks_detected']} R-peaks")
            print(f"  - Extracted {lead_beats['valid_beats_extracted']} valid beats")
            if lead_beats['heart_rate_bpm']:
                print(f"  - Estimated heart rate: {lead_beats['heart_rate_bpm']:.1f} BPM")
        
        # Save extracted beats
        if output_file is None:
            base_name = os.path.splitext(json_file)[0]
            output_file = f"{base_name}_beats.json"
        
        print(f"Saving extracted beats to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(extracted_data, f, indent=2)
        
        print(f"Beat extraction completed!")
        print(f"Output file: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
        return extracted_data

def visualize_beat_extraction(json_file: str, lead_name: str = 'II', 
                            max_beats_to_show: int = 5, save_plot: bool = True):
    """
    Visualize the beat extraction results
    """
    extractor = ECGBeatExtractor()
    
    # Load original ECG data
    with open(json_file, 'r') as f:
        ecg_data = json.load(f)
    
    # Extract beats
    extracted_data = extractor.extract_beats_from_json(json_file, leads_to_process=[lead_name])
    
    # Get the lead data
    original_signal = np.array(ecg_data['leads'][lead_name])
    lead_beats = extracted_data['leads'][lead_name]
    r_peaks = np.array(lead_beats['r_peak_indices'])
    
    # Create time axis
    sampling_rate = ecg_data['metadata']['sampling_rate']
    time_axis = np.arange(len(original_signal)) / sampling_rate
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Original signal with detected R-peaks
    axes[0].plot(time_axis, original_signal, 'b-', linewidth=0.8, label=f'Lead {lead_name}')
    axes[0].plot(r_peaks / sampling_rate, original_signal[r_peaks], 'ro', 
                markersize=6, label='Detected R-peaks')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude (mV)')
    axes[0].set_title(f'ECG Signal with Detected R-peaks - Lead {lead_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Individual beat segments
    beats_to_show = min(max_beats_to_show, len(lead_beats['beats']))
    beat_time_axis = np.arange(extractor.beat_length) / sampling_rate * 1000 - \
                     extractor.pre_r_samples / sampling_rate * 1000
    
    colors = plt.cm.tab10(np.linspace(0, 1, beats_to_show))
    for i in range(beats_to_show):
        beat_data = np.array(lead_beats['beats'][i]['beat_segment'])
        axes[1].plot(beat_time_axis, beat_data, color=colors[i], 
                    linewidth=1.5, label=f'Beat {i+1}')
    
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='R-peak')
    axes[1].set_xlabel('Time relative to R-peak (ms)')
    axes[1].set_ylabel('Amplitude (mV)')
    axes[1].set_title(f'Individual Beat Segments - Lead {lead_name}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plot_filename = f"{os.path.splitext(json_file)[0]}_beat_extraction_visualization.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved as: {plot_filename}")
    
    plt.show()

def main():
    """
    Main function to demonstrate beat extraction
    """
    # Configuration
    input_json_file = "china_ecg_complete_data_954064495204269526041_2019-04-26154016.json"
    
    # Check if input file exists
    if not os.path.exists(input_json_file):
        print(f"Error: Input file '{input_json_file}' not found!")
        print("Please make sure the JSON file from your conversion script exists.")
        return
    
    # Create beat extractor
    extractor = ECGBeatExtractor(sampling_rate=500)
    
    # Extract beats from all leads
    print("=== ECG Beat Extraction ===")
    extracted_data = extractor.extract_beats_from_json(
        input_json_file,
        leads_to_process=['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']  # Focus on measured leads
    )
    
    # Print summary
    print("\n=== Extraction Summary ===")
    for lead_name, lead_info in extracted_data['leads'].items():
        print(f"Lead {lead_name}:")
        print(f"  - R-peaks detected: {lead_info['total_r_peaks_detected']}")
        print(f"  - Valid beats extracted: {lead_info['valid_beats_extracted']}")
        if lead_info['heart_rate_bpm']:
            print(f"  - Heart rate: {lead_info['heart_rate_bpm']:.1f} BPM")
        print()
    
    # Create visualization for Lead II (commonly used for rhythm analysis)
    print("Creating visualization...")
    try:
        visualize_beat_extraction(input_json_file, lead_name='II', max_beats_to_show=5)
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("Beat extraction completed successfully despite visualization error.")

if __name__ == "__main__":
    main()
