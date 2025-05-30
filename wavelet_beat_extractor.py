import json
import numpy as np
import os
import argparse
from scipy import signal
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pywt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class WaveletDenoiser:
    """
    Wavelet-based ECG signal denoising using adaptive thresholding.
    """
    
    def __init__(self, wavelet='db6', levels=6, threshold_mode='soft'):
        self.wavelet = wavelet
        self.levels = levels
        self.threshold_mode = threshold_mode
        
    def denoise_signal(self, signal_data):
        """
        Denoise ECG signal using wavelet decomposition.
        
        Args:
            signal_data: Input ECG signal
            
        Returns:
            denoised_signal: Denoised ECG signal
            noise_removed: Estimated noise component
        """
        # Wavelet decomposition
        coeffs = pywt.wavedec(signal_data, self.wavelet, level=self.levels)
        
        # Estimate noise standard deviation using MAD (Median Absolute Deviation)
        # from the finest detail coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Adaptive thresholding for each level
        threshold_coeffs = list(coeffs)
        
        for i in range(1, len(coeffs)):  # Skip approximation coefficients
            # Calculate threshold for this level
            # Higher levels (coarser details) get lower thresholds
            level_factor = 1.0 / np.sqrt(2 ** (len(coeffs) - i))
            threshold = sigma * np.sqrt(2 * np.log(len(signal_data))) * level_factor
            
            # Apply thresholding
            threshold_coeffs[i] = pywt.threshold(coeffs[i], threshold, 
                                               mode=self.threshold_mode)
        
        # Reconstruct denoised signal
        denoised_signal = pywt.waverec(threshold_coeffs, self.wavelet)
        
        # Calculate removed noise
        noise_removed = signal_data - denoised_signal
        
        return denoised_signal, noise_removed

class EnhancedPanTompkinsDetector:
    """
    Enhanced Pan-Tompkins QRS detection with wavelet preprocessing.
    """
    
    def __init__(self, sampling_rate=500):
        self.fs = sampling_rate
        self.denoiser = WaveletDenoiser()
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
        """Find peaks using adaptive thresholding."""
        if min_distance is None:
            min_distance = int(0.2 * self.fs)
        
        # Initialize thresholds
        spki = 0  # Signal peak
        npki = 0  # Noise peak
        
        # Simple peak detection for initialization
        initial_peaks, _ = signal.find_peaks(signal_data, 
                                           height=np.max(signal_data) * 0.3,
                                           distance=min_distance)
        
        if len(initial_peaks) < 2:
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
        peaks = []
        rr_intervals = []
        
        for i, peak_idx in enumerate(initial_peaks):
            peak_height = signal_data[peak_idx]
            
            if peak_height > threshold1:
                peaks.append(peak_idx)
                spki = 0.125 * peak_height + 0.875 * spki
                
                if len(peaks) > 1:
                    rr_interval = peaks[-1] - peaks[-2]
                    rr_intervals.append(rr_interval)
            else:
                npki = 0.125 * peak_height + 0.875 * npki
            
            threshold1 = npki + 0.25 * (spki - npki)
            threshold2 = 0.5 * threshold1
        
        return np.array(peaks), np.array(rr_intervals)
    
    def detect_qrs(self, ecg_signal, apply_denoising=True):
        """Complete enhanced Pan-Tompkins QRS detection pipeline."""
        
        # Step 0: Wavelet denoising (optional)
        if apply_denoising:
            denoised_signal, noise_removed = self.denoiser.denoise_signal(ecg_signal)
            processing_signal = denoised_signal
        else:
            processing_signal = ecg_signal
            denoised_signal = ecg_signal
            noise_removed = np.zeros_like(ecg_signal)
        
        # Step 1: Bandpass filter (5-15 Hz)
        filtered = self.bandpass_filter(processing_signal)
        
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
            'denoised_signal': denoised_signal,
            'noise_removed': noise_removed,
            'filtered_signal': filtered,
            'derivative': derivative,
            'squared': squared,
            'integrated': integrated
        }

class TemplateMatching:
    """
    Template matching for QRS complex verification and refinement.
    """
    
    def __init__(self, sampling_rate=500):
        self.fs = sampling_rate
        self.templates = []
        self.template_labels = []
        
    def extract_beat_segments(self, ecg_signal, peak_indices, window_ms=300):
        """
        Extract beat segments around detected peaks.
        
        Args:
            ecg_signal: ECG signal
            peak_indices: Detected peak locations
            window_ms: Window size around peaks in milliseconds
            
        Returns:
            beat_segments: List of beat segments
            valid_indices: Indices of valid peaks
        """
        window_samples = int(window_ms * self.fs / 1000)
        half_window = window_samples // 2
        
        beat_segments = []
        valid_indices = []
        
        for i, peak_idx in enumerate(peak_indices):
            start_idx = peak_idx - half_window
            end_idx = peak_idx + half_window
            
            # Check if segment is within signal bounds
            if start_idx >= 0 and end_idx < len(ecg_signal):
                segment = ecg_signal[start_idx:end_idx]
                beat_segments.append(segment)
                valid_indices.append(i)
        
        return beat_segments, valid_indices
    
    def create_templates(self, beat_segments, n_templates=3, method='kmeans'):
        """
        Create beat templates using clustering.
        
        Args:
            beat_segments: List of beat segments
            n_templates: Number of templates to create
            method: Clustering method ('kmeans' or 'median')
            
        Returns:
            templates: List of template waveforms
            labels: Cluster labels for each beat
        """
        if len(beat_segments) < n_templates:
            # If too few beats, use all as templates
            self.templates = beat_segments
            self.template_labels = list(range(len(beat_segments)))
            return self.templates, self.template_labels
        
        # Normalize beat segments
        normalized_beats = []
        for beat in beat_segments:
            # Normalize to zero mean and unit variance
            normalized_beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
            normalized_beats.append(normalized_beat)
        
        normalized_beats = np.array(normalized_beats)
        
        if method == 'kmeans':
            # K-means clustering
            kmeans = KMeans(n_clusters=n_templates, random_state=42, n_init=10)
            labels = kmeans.fit_predict(normalized_beats)
            
            # Create templates as cluster centroids
            templates = []
            for i in range(n_templates):
                cluster_beats = normalized_beats[labels == i]
                if len(cluster_beats) > 0:
                    template = np.mean(cluster_beats, axis=0)
                    templates.append(template)
                else:
                    # If empty cluster, use a random beat
                    templates.append(normalized_beats[0])
            
        elif method == 'median':
            # Simple median-based template (use first few beats)
            templates = normalized_beats[:n_templates]
            labels = np.arange(len(beat_segments)) % n_templates
        
        self.templates = templates
        self.template_labels = labels
        
        return templates, labels
    
    def match_templates(self, beat_segments, correlation_threshold=0.7):
        """
        Match beat segments against templates.
        
        Args:
            beat_segments: List of beat segments to match
            correlation_threshold: Minimum correlation for valid match
            
        Returns:
            matches: List of template matches for each beat
            correlations: Correlation values
            valid_beats: Boolean array indicating valid beats
        """
        if not self.templates:
            raise ValueError("No templates created. Call create_templates first.")
        
        matches = []
        correlations = []
        valid_beats = []
        
        for beat in beat_segments:
            # Normalize beat
            normalized_beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
            
            # Calculate correlation with each template
            max_corr = -1
            best_match = -1
            
            for i, template in enumerate(self.templates):
                corr, _ = pearsonr(normalized_beat, template)
                if corr > max_corr:
                    max_corr = corr
                    best_match = i
            
            matches.append(best_match)
            correlations.append(max_corr)
            valid_beats.append(max_corr >= correlation_threshold)
        
        return matches, correlations, np.array(valid_beats)
    
    def refine_peak_locations(self, ecg_signal, peak_indices, beat_segments, 
                            search_window_ms=50):
        """
        Refine peak locations using template matching.
        
        Args:
            ecg_signal: Original ECG signal
            peak_indices: Initial peak locations
            beat_segments: Extracted beat segments
            search_window_ms: Search window for peak refinement
            
        Returns:
            refined_peaks: Refined peak locations
        """
        search_samples = int(search_window_ms * self.fs / 1000)
        half_search = search_samples // 2
        
        refined_peaks = []
        
        for i, peak_idx in enumerate(peak_indices):
            if i < len(beat_segments):
                # Search for best match in local neighborhood
                start_search = max(0, peak_idx - half_search)
                end_search = min(len(ecg_signal), peak_idx + half_search)
                
                best_corr = -1
                best_peak = peak_idx
                
                # Try different positions within search window
                for test_peak in range(start_search, end_search):
                    # Extract segment around test peak
                    segment_start = max(0, test_peak - len(beat_segments[i])//2)
                    segment_end = min(len(ecg_signal), test_peak + len(beat_segments[i])//2)
                    
                    if segment_end - segment_start == len(beat_segments[i]):
                        test_segment = ecg_signal[segment_start:segment_end]
                        
                        # Calculate correlation with original beat
                        corr, _ = pearsonr(test_segment, beat_segments[i])
                        
                        if corr > best_corr:
                            best_corr = corr
                            best_peak = test_peak
                
                refined_peaks.append(best_peak)
            else:
                refined_peaks.append(peak_idx)
        
        return np.array(refined_peaks)

def extract_beats_wavelet_enhanced(json_file, output_folder, 
                                 lead_name='II', 
                                 beat_window_ms=600, 
                                 min_rr_ms=300, 
                                 max_rr_ms=2000,
                                 apply_denoising=True,
                                 correlation_threshold=0.7,
                                 n_templates=3,
                                 plot_results=False):
    """
    Extract ECG beats using wavelet-enhanced approach.
    
    Args:
        json_file: Path to ECG JSON file
        output_folder: Output folder for beat files
        lead_name: ECG lead to use for beat detection
        beat_window_ms: Window size around each beat (milliseconds)
        min_rr_ms: Minimum RR interval (milliseconds)
        max_rr_ms: Maximum RR interval (milliseconds)
        apply_denoising: Whether to apply wavelet denoising
        correlation_threshold: Minimum correlation for template matching
        n_templates: Number of templates to create
        plot_results: Whether to plot results
    """
    
    print(f"Processing {json_file} with wavelet-enhanced approach...")
    
    # Load ECG data
    with open(json_file, 'r') as f:
        ecg_data = json.load(f)
    
    # Extract metadata
    if 'metadata' in ecg_data:
        metadata = ecg_data['metadata']
        leads = ecg_data['leads']
        sampling_rate = metadata['sampling_rate']
    else:
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
    
    # Initialize enhanced detector
    detector = EnhancedPanTompkinsDetector(sampling_rate)
    
    # Step 1: Enhanced Pan-Tompkins detection
    print("Running enhanced Pan-Tompkins QRS detection...")
    detection_results = detector.detect_qrs(ecg_signal, apply_denoising=apply_denoising)
    initial_peaks = detection_results['peaks']
    
    print(f"Initial detection found {len(initial_peaks)} potential QRS complexes")
    
    # Step 2: Filter peaks based on RR interval constraints
    min_rr_samples = int(min_rr_ms * sampling_rate / 1000)
    max_rr_samples = int(max_rr_ms * sampling_rate / 1000)
    
    filtered_peaks = []
    for i, peak in enumerate(initial_peaks):
        if i == 0:
            filtered_peaks.append(peak)
        else:
            rr_interval = peak - initial_peaks[i-1]
            if min_rr_samples <= rr_interval <= max_rr_samples:
                filtered_peaks.append(peak)
    
    filtered_peaks = np.array(filtered_peaks)
    print(f"After RR interval filtering: {len(filtered_peaks)} peaks")
    
    if len(filtered_peaks) < 3:
        print("Too few peaks for template matching. Using basic detection.")
        final_peaks = filtered_peaks
        template_correlations = np.ones(len(filtered_peaks))
        valid_template_matches = np.ones(len(filtered_peaks), dtype=bool)
    else:
        # Step 3: Template matching
        print("Performing template matching...")
        template_matcher = TemplateMatching(sampling_rate)
        
        # Extract beat segments for template creation
        beat_segments, valid_indices = template_matcher.extract_beat_segments(
            detection_results['denoised_signal'], filtered_peaks, window_ms=300
        )
        
        print(f"Extracted {len(beat_segments)} beat segments for template analysis")
        
        # Create templates
        templates, template_labels = template_matcher.create_templates(
            beat_segments, n_templates=n_templates, method='kmeans'
        )
        
        print(f"Created {len(templates)} beat templates")
        
        # Match beats against templates
        matches, correlations, valid_beats = template_matcher.match_templates(
            beat_segments, correlation_threshold=correlation_threshold
        )
        
        # Filter peaks based on template matching
        valid_peak_indices = np.array(valid_indices)[valid_beats]
        final_peaks = filtered_peaks[valid_peak_indices]
        template_correlations = np.array(correlations)[valid_beats]
        valid_template_matches = valid_beats
        
        print(f"Template matching validation: {len(final_peaks)} valid beats (correlation >= {correlation_threshold})")
        
        # Step 4: Refine peak locations
        print("Refining peak locations...")
        valid_beat_segments = [beat_segments[i] for i in range(len(beat_segments)) if valid_beats[i]]
        refined_peaks = template_matcher.refine_peak_locations(
            ecg_signal, final_peaks, valid_beat_segments, search_window_ms=50
        )
        
        final_peaks = refined_peaks
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Calculate beat window in samples
    beat_window_samples = int(beat_window_ms * sampling_rate / 1000)
    half_window = beat_window_samples // 2
    
    # Extract individual beats
    beats_data = []
    valid_beat_count = 0
    prefix = os.path.splitext(os.path.basename(json_file))[0]
    
    for i, peak_idx in enumerate(final_peaks):
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
        
        # Add template matching info
        if i < len(template_correlations):
            beat_metadata['template_correlation'] = float(template_correlations[i])
        
        # Add RR interval info
        if i > 0:
            rr_interval_samples = peak_idx - final_peaks[i-1]
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
    
    # Save enhanced summary file
    summary_data = {
        'source_file': json_file,
        'processing_parameters': {
            'detection_lead': lead_name,
            'beat_window_ms': beat_window_ms,
            'min_rr_ms': min_rr_ms,
            'max_rr_ms': max_rr_ms,
            'sampling_rate': sampling_rate,
            'wavelet_denoising': apply_denoising,
            'correlation_threshold': correlation_threshold,
            'n_templates': n_templates
        },
        'results': {
            'initial_peaks_detected': len(initial_peaks),
            'rr_filtered_peaks': len(filtered_peaks),
            'template_validated_beats': valid_beat_count,
            'average_template_correlation': float(np.mean(template_correlations)) if len(template_correlations) > 0 else None,
            'detection_stages': {
                'pan_tompkins': len(initial_peaks),
                'rr_filtering': len(filtered_peaks),
                'template_matching': valid_beat_count
            }
        },
        'beat_files': [f"{prefix}_beat_{i:04d}.json" for i in range(valid_beat_count)]
    }
    
    if len(final_peaks) > 1:
        rr_intervals = np.diff(final_peaks) * 1000 / sampling_rate
        summary_data['results']['average_rr_interval_ms'] = float(np.mean(rr_intervals))
        summary_data['results']['average_heart_rate_bpm'] = 60000 / np.mean(rr_intervals)
    
    summary_filename = f"{output_folder}/{prefix}_enhanced_beats_summary.json"
    with open(summary_filename, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Plot results if requested
    if plot_results:
        plot_wavelet_enhanced_results(ecg_signal, detection_results, final_peaks, 
                                    template_correlations, sampling_rate, 
                                    output_folder, lead_name, apply_denoising)
    
    return beats_data

def plot_wavelet_enhanced_results(ecg_signal, detection_results, final_peaks, 
                                template_correlations, sampling_rate, 
                                output_folder, lead_name, apply_denoising):
    """Plot wavelet-enhanced detection results."""
    
    time_axis = np.arange(len(ecg_signal)) / sampling_rate
    
    n_plots = 6 if apply_denoising else 5
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4*n_plots))
    fig.suptitle(f'Wavelet-Enhanced QRS Detection Results - Lead {lead_name}', fontsize=16)
    
    plot_idx = 0
    
    # Original signal
    axes[plot_idx].plot(time_axis, ecg_signal, 'b-', linewidth=0.8, label='Original')
    axes[plot_idx].plot(time_axis[final_peaks], ecg_signal[final_peaks], 'ro', markersize=4, label='Detected Beats')
    axes[plot_idx].set_title('Original ECG Signal with Final Detected Beats')
    axes[plot_idx].set_ylabel('Amplitude (mV)')
    axes[plot_idx].legend()
    axes[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1
    
    # Denoised signal (if denoising was applied)
    if apply_denoising:
        axes[plot_idx].plot(time_axis, detection_results['denoised_signal'], 'g-', linewidth=0.8, label='Denoised')
        axes[plot_idx].plot(time_axis, detection_results['noise_removed'], 'r-', linewidth=0.5, alpha=0.7, label='Removed Noise')
        axes[plot_idx].set_title('Wavelet Denoised Signal and Removed Noise')
        axes[plot_idx].set_ylabel('Amplitude')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Filtered signal
    axes[plot_idx].plot(time_axis, detection_results['filtered_signal'], 'orange', linewidth=0.8)
    axes[plot_idx].set_title('Bandpass Filtered (5-15 Hz)')
    axes[plot_idx].set_ylabel('Amplitude')
    axes[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1
    
    # Derivative
    axes[plot_idx].plot(time_axis, detection_results['derivative'], 'purple', linewidth=0.8)
    axes[plot_idx].set_title('Derivative Filter Output')
    axes[plot_idx].set_ylabel('Amplitude')
    axes[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1
    
    # Squared
    axes[plot_idx].plot(time_axis, detection_results['squared'], 'brown', linewidth=0.8)
    axes[plot_idx].set_title('Squared Signal')
    axes[plot_idx].set_ylabel('Amplitude')
    axes[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1
    
    # Integrated with peaks and template correlations
    axes[plot_idx].plot(time_axis, detection_results['integrated'], 'r-', linewidth=0.8, label='Integrated Signal')
    axes[plot_idx].plot(time_axis[final_peaks], detection_results['integrated'][final_peaks], 'ko', markersize=4, label='Final Peaks')
    
    # Color-code peaks by template correlation
    if len(template_correlations) > 0:
        scatter = axes[plot_idx].scatter(time_axis[final_peaks[:len(template_correlations)]], 
                                       detection_results['integrated'][final_peaks[:len(template_correlations)]], 
                                       c=template_correlations, cmap='viridis', s=50, alpha=0.8)
        plt.colorbar(scatter, ax=axes[plot_idx], label='Template Correlation')
    
    axes[plot_idx].set_title('Moving Window Integration with Template-Validated Peaks')
    axes[plot_idx].set_ylabel('Amplitude')
    axes[plot_idx].set_xlabel('Time (seconds)')
    axes[plot_idx].legend()
    axes[plot_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{output_folder}/wavelet_enhanced_results.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced detection plot saved to {plot_filename}")

def main():
    parser = argparse.ArgumentParser(description="Extract ECG beats using wavelet-enhanced approach")
    parser.add_argument("input_file", help="Input ECG JSON file")
    parser.add_argument("-o", "--output", default="extracted_beats", 
                       help="Output folder for beat files (default: extracted_beats)")
    parser.add_argument("--lead", default="II", 
                       help="ECG lead to use for beat detection (default: II)")
    parser.add_argument("--window", type=int, default=600,
                       help="Beat window size in milliseconds (default: 600)")
    parser.add_argument("--min-rr", type=int, default=300,
                       help="Minimum RR interval in milliseconds (default: 300)")
    parser.add_argument("--max-rr", type=int, default=2000,
                       help="Maximum RR interval in milliseconds (default: 2000)")
    parser.add_argument("--no-denoising", action="store_true",
                       help="Disable wavelet denoising")
    parser.add_argument("--correlation-threshold", type=float, default=0.7,
                       help="Template correlation threshold (default: 0.7)")
    parser.add_argument("--templates", type=int, default=3,
                       help="Number of templates to create (default: 3)")
    parser.add_argument("--plot", action="store_true",
                       help="Generate result plots")
    parser.add_argument("--batch", action="store_true",
                       help="Process all JSON files in input directory")
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch processing mode
        if not os.path.isdir(args.input_file):
            print(f"Error: {args.input_file} is not a directory for batch processing")
            return
        
        json_files = [f for f in os.listdir(args.input_file) if f.endswith('.json')]
        if not json_files:
            print(f"No JSON files found in {args.input_file}")
            return
        
        print(f"Found {len(json_files)} JSON files for batch processing")
        
        for json_file in json_files:
            input_path = os.path.join(args.input_file, json_file)
            output_folder = os.path.join(args.output, os.path.splitext(json_file)[0])
            
            try:
                extract_beats_wavelet_enhanced(
                    json_file=input_path,
                    output_folder=output_folder,
                    lead_name=args.lead,
                    beat_window_ms=args.window,
                    min_rr_ms=args.min_rr,
                    max_rr_ms=args.max_rr,
                    apply_denoising=not args.no_denoising,
                    correlation_threshold=args.correlation_threshold,
                    n_templates=args.templates,
                    plot_results=args.plot
                )
                print(f"Successfully processed {json_file}")
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
    
    else:
        # Single file processing
        if not os.path.isfile(args.input_file):
            print(f"Error: {args.input_file} not found")
            return
        
        try:
            extract_beats_wavelet_enhanced(
                json_file=args.input_file,
                output_folder=args.output,
                lead_name=args.lead,
                beat_window_ms=args.window,
                min_rr_ms=args.min_rr,
                max_rr_ms=args.max_rr,
                apply_denoising=not args.no_denoising,
                correlation_threshold=args.correlation_threshold,
                n_templates=args.templates,
                plot_results=args.plot
            )
            print("Processing completed successfully!")
        except Exception as e:
            print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()