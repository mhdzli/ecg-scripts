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
                 min_lead_agreement=1, tolerance_ms=100, refine_factor=None):
        self.method = method
        self.extraction_method = extraction_method
        self.window_size = window_size
        self.min_lead_agreement = min_lead_agreement
        self.tolerance_ms = tolerance_ms
        self.refine_factor = refine_factor
        
        # Validate method
        valid_methods = ['simple', 'pan_tompkins', 'hamilton_tompkins', 'wavelet', 'christov', 'ensemble', 'pan_tompkins_simple', 'simple_enhanced']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of: {valid_methods}")

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
    
    def detect_r_peaks_simple_enhanced(self, ecg_signal, fs):
        """Enhanced simple R-peak detection with better preprocessing"""
        
        # Remove baseline wander with high-pass filter
        filtered = self.preprocess_ecg(ecg_signal, fs, lowcut=1, highcut=45)
        
        # Use absolute value to handle inverted leads
        abs_signal = np.abs(filtered)
        
        # Adaptive threshold based on signal statistics
        signal_mean = np.mean(abs_signal)
        signal_std = np.std(abs_signal)
        
        # Use robust statistics
        signal_median = np.median(abs_signal)
        signal_mad = np.median(np.abs(abs_signal - signal_median))  # Median Absolute Deviation
        
        # Set threshold using median and MAD (more robust than mean/std)
        threshold = signal_median + 2.5 * signal_mad
        
        # Minimum distance between peaks
        min_distance = max(1, int(0.3 * fs))  # 300ms
        
        # Find peaks
        peaks, properties = find_peaks(abs_signal,
                                      height=threshold,
                                      distance=min_distance,
                                      prominence=signal_mad)
        
        # Additional validation: remove peaks that are too close to signal edges
        edge_margin = int(0.1 * fs)  # 100ms margin
        valid_peaks = peaks[(peaks >= edge_margin) & (peaks < len(ecg_signal) - edge_margin)]
        
        return valid_peaks

    def detect_r_peaks_pan_tompkins_simple(self, ecg_signal, fs):
        """Simplified Pan-Tompkins focusing on reliability"""
        
        # Preprocessing with gentler filtering
        filtered = self.preprocess_ecg(ecg_signal, fs, lowcut=3, highcut=45)
        
        # Simple derivative
        derivative = np.diff(filtered, prepend=filtered[0])
        
        # Squaring
        squared = derivative ** 2
        
        # Moving average integration
        window_size = max(1, int(0.1 * fs))  # 100ms window
        kernel = np.ones(window_size) / window_size
        integrated = np.convolve(squared, kernel, mode='same')
        
        # Simple peak detection
        # Use a more conservative approach
        threshold = np.mean(integrated) + 1.5 * np.std(integrated)
        min_distance = max(1, int(0.3 * fs))  # 300ms minimum distance
        
        peaks, _ = find_peaks(integrated, 
                             height=threshold, 
                             distance=min_distance)
        
        # Refine to original signal
        refined_peaks = []
        search_radius = max(1, int(0.04 * fs))  # 40ms search
        
        for peak in peaks:
            start = max(0, peak - search_radius)
            end = min(len(ecg_signal), peak + search_radius)
            
            # Find max absolute value in original signal
            segment = np.abs(ecg_signal[start:end])
            if len(segment) > 0:
                max_idx = np.argmax(segment)
                refined_peak = start + max_idx
                refined_peaks.append(refined_peak)
        
        return np.array(refined_peaks)

    def detect_r_peaks_pan_tompkins(self, ecg_signal, fs):
        """Improved Pan-Tompkins algorithm with better preprocessing and thresholding"""
        
        # Step 1: Bandpass filter (5-15 Hz)
        nyquist = 0.5 * fs
        low = 5.0 / nyquist
        high = 15.0 / nyquist
        
        # Ensure frequencies are valid
        if high >= 1.0:
            high = 0.99
        if low <= 0:
            low = 0.01
        
        try:
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, ecg_signal)
        except:
            # Fallback if filter fails
            filtered = ecg_signal.copy()
        
        # Step 2: Derivative (5-point derivative approximation)
        # Pad signal to handle edges
        padded = np.pad(filtered, 2, mode='edge')
        derivative = np.zeros_like(filtered)
        
        for i in range(len(filtered)):
            derivative[i] = (2*padded[i+4] + padded[i+3] - padded[i+1] - 2*padded[i]) / 8.0
        
        # Step 3: Squaring
        squared = derivative ** 2
        
        # Step 4: Moving window integration (150ms window)
        window_samples = max(1, int(0.150 * fs))
        integrated = np.convolve(squared, np.ones(window_samples)/window_samples, mode='same')
        
        # Step 5: Find peaks in integrated signal with proper thresholding
        # Initial peak detection to establish thresholds
        initial_threshold = 0.2 * np.max(integrated)
        min_distance = max(1, int(0.2 * fs))  # 200ms minimum distance
        
        candidate_peaks, _ = find_peaks(integrated, 
                                       height=initial_threshold,
                                       distance=min_distance)
        
        if len(candidate_peaks) == 0:
            # Lower threshold if no peaks found
            initial_threshold = 0.1 * np.max(integrated)
            candidate_peaks, _ = find_peaks(integrated, 
                                           height=initial_threshold,
                                           distance=min_distance)
        
        if len(candidate_peaks) == 0:
            return np.array([])
        
        # Step 6: Adaptive thresholding based on detected peaks
        peak_values = integrated[candidate_peaks]
        
        # Calculate noise floor (use percentile instead of mean to be robust)
        noise_level = np.percentile(integrated, 25)
        signal_level = np.percentile(peak_values, 75)
        
        # Set adaptive threshold
        adaptive_threshold = noise_level + 0.25 * (signal_level - noise_level)
        
        # Final peak detection with adaptive threshold
        final_peaks, properties = find_peaks(integrated,
                                            height=adaptive_threshold,
                                            distance=min_distance,
                                            prominence=0.1 * signal_level)
        
        if len(final_peaks) == 0:
            return np.array([])
        
        # Step 7: Refine peak locations to original signal
        refined_peaks = []
        search_window = max(1, int(0.05 * fs))  # 50ms search window
        
        for peak_idx in final_peaks:
            # Search around the detected peak in the original signal
            start_search = max(0, peak_idx - search_window)
            end_search = min(len(ecg_signal), peak_idx + search_window)
            
            if end_search > start_search:
                # Find the maximum absolute value in the search window
                search_segment = np.abs(ecg_signal[start_search:end_search])
                local_peak_idx = np.argmax(search_segment)
                refined_peak = start_search + local_peak_idx
                
                # Validate the refined peak
                if 0 <= refined_peak < len(ecg_signal):
                    refined_peaks.append(refined_peak)
        
        return np.array(refined_peaks)
    
    def detect_r_peaks_wavelet(self, ecg_signal, fs):
        """Wavelet-based R-peak detection"""
        try:
            import pywt
        except ImportError:
            print("PyWavelets not installed, falling back to simple method")
            return self.detect_r_peaks_simple(ecg_signal, fs)
        
        # Wavelet transform
        scales = np.arange(1, 32)
        coeffs, freqs = pywt.cwt(ecg_signal, scales, 'mexh', 1/fs)
        
        # Find scale corresponding to QRS frequency (around 10-15 Hz)
        qrs_freq = 12  # Hz
        qrs_scale_idx = np.argmin(np.abs(freqs - qrs_freq))
        qrs_coeffs = coeffs[qrs_scale_idx]
        
        # Peak detection on wavelet coefficients
        threshold = 0.3 * np.max(np.abs(qrs_coeffs))
        min_distance = int(0.3 * fs)
        
        peaks, _ = find_peaks(np.abs(qrs_coeffs), height=threshold, distance=min_distance)
        
        # Refine to original signal
        refined_peaks = []
        search_window = int(0.05 * fs)
        
        for peak in peaks:
            start = max(0, peak - search_window)
            end = min(len(ecg_signal), peak + search_window)
            local_max = start + np.argmax(np.abs(ecg_signal[start:end]))
            refined_peaks.append(local_max)
        
        return np.array(refined_peaks)

    def detect_r_peaks_ensemble(self, ecg_signal, fs):
        """Ensemble method combining multiple algorithms"""
        methods = []
        
        # Run different methods
        try:
            peaks_simple = self.detect_r_peaks_simple(ecg_signal, fs)
            methods.append(('simple', peaks_simple))
        except: pass
        
        try:
            peaks_pt = self.detect_r_peaks_pan_tompkins(ecg_signal, fs)
            methods.append(('pan_tompkins', peaks_pt))
        except: pass
        
        try:
            peaks_hamilton = self.detect_r_peaks_hamilton_tompkins(ecg_signal, fs)
            methods.append(('hamilton', peaks_hamilton))
        except: pass
        
        try:
            peaks_christov = self.detect_r_peaks_christov(ecg_signal, fs)
            methods.append(('christov', peaks_christov))
        except: pass
        
        if not methods:
            return np.array([])
        
        # Combine results using voting
        all_peaks = []
        for name, peaks in methods:
            all_peaks.extend(peaks)
        
        if not all_peaks:
            return np.array([])
        
        # Cluster nearby peaks
        tolerance = int(0.05 * fs)  # 50ms tolerance
        clustered_peaks = []
        all_peaks.sort()
        
        current_cluster = [all_peaks[0]]
        for peak in all_peaks[1:]:
            if peak - current_cluster[-1] <= tolerance:
                current_cluster.append(peak)
            else:
                # Take median of cluster
                clustered_peaks.append(int(np.median(current_cluster)))
                current_cluster = [peak]
        
        # Don't forget the last cluster
        if current_cluster:
            clustered_peaks.append(int(np.median(current_cluster)))
        
        return np.array(clustered_peaks)

    def detect_r_peaks_christov(self, ecg_signal, fs):
        """Christov QRS detection algorithm"""
        # High-pass filter
        filtered = self.preprocess_ecg(ecg_signal, fs, 5, 15)
        
        # Moving average subtraction (removes baseline drift)
        ma_window = int(0.2 * fs)  # 200ms
        ma = np.convolve(filtered, np.ones(ma_window)/ma_window, mode='same')
        filtered = filtered - ma
        
        # Complex lead formation (if multiple leads available)
        # For single lead, use derivative
        derivative = np.gradient(filtered)
        
        # K-factor (combination of signal and derivative)
        k_factor = filtered + derivative
        
        # Squaring
        squared = k_factor ** 2
        
        # Moving integration (66ms window)
        int_window = int(0.066 * fs)
        integrated = np.convolve(squared, np.ones(int_window)/int_window, mode='same')
        
        # Adaptive threshold
        threshold = 0.4 * np.max(integrated)
        min_distance = int(0.25 * fs)  # 250ms
        
        peaks, _ = find_peaks(integrated, height=threshold, distance=min_distance)
        
        # Refine peaks
        refined_peaks = []
        search_window = int(0.04 * fs)
        
        for peak in peaks:
            start = max(0, peak - search_window)
            end = min(len(ecg_signal), peak + search_window)
            local_max = start + np.argmax(np.abs(ecg_signal[start:end]))
            refined_peaks.append(local_max)
        
        return np.array(refined_peaks)
    def detect_r_peaks_hamilton_tompkins(self, ecg_signal, fs):
        """Hamilton-Tompkins QRS detection algorithm"""
        # Preprocessing
        filtered = self.preprocess_ecg(ecg_signal, fs, 8, 16)
        
        # First derivative
        derivative = np.gradient(filtered)
        
        # Squaring
        squared = derivative ** 2
        
        # Moving average (80ms window)
        window_size = int(0.08 * fs)
        ma_filter = np.ones(window_size) / window_size
        ma_signal = np.convolve(squared, ma_filter, mode='same')
        
        # Peak detection with adaptive threshold
        threshold = 0.3 * np.max(ma_signal)
        min_distance = int(0.3 * fs)  # 300ms minimum distance
        
        peaks, _ = find_peaks(ma_signal, height=threshold, distance=min_distance)
        
        # Refine to original signal
        refined_peaks = []
        search_window = int(0.04 * fs)  # 40ms search window
        
        for peak in peaks:
            start = max(0, peak - search_window)
            end = min(len(ecg_signal), peak + search_window)
            local_segment = np.abs(ecg_signal[start:end])
            if len(local_segment) > 0:
                local_max = start + np.argmax(local_segment)
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
    
    def refine_r_peaks(self, signal, r_peaks, fs):
        """
        Refine R-peak locations by searching for maximum values in intervals around detected peaks
        """
        if self.refine_factor is None or len(r_peaks) < 2:
            return r_peaks
        
        refined_peaks = []
        
        for i, peak in enumerate(r_peaks):
            # Calculate interval size for this peak
            if i == 0:
                # For first peak, use interval to next peak
                if len(r_peaks) > 1:
                    interval = r_peaks[1] - r_peaks[0]
                else:
                    interval = int(0.8 * fs)  # Default 800ms
            elif i == len(r_peaks) - 1:
                # For last peak, use interval from previous peak
                interval = r_peaks[i] - r_peaks[i-1]
            else:
                # For middle peaks, use average of surrounding intervals
                prev_interval = r_peaks[i] - r_peaks[i-1]
                next_interval = r_peaks[i+1] - r_peaks[i]
                interval = (prev_interval + next_interval) / 2
            
            # Calculate search window based on interval factor
            search_radius = int(interval * self.refine_factor)
            
            # Define search window
            start_search = max(0, peak - search_radius)
            end_search = min(len(signal), peak + search_radius)
            
            # Find maximum absolute value in the search window
            if end_search > start_search:
                search_segment = signal[start_search:end_search]
                max_idx = np.argmax(search_segment)
                refined_peak = start_search + max_idx
                refined_peaks.append(refined_peak)
            else:
                # If search window is invalid, keep original peak
                refined_peaks.append(peak)
        
        return np.array(refined_peaks)

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
        first_start = max((r_peaks[0] - round((r_peaks[1] - r_peaks[0]) * 4.5 / 10)), 0)
        if first_end <= signal_length:
            beat_data = {}
            for lead_name, signal in leads.items():
                beat_data[lead_name] = signal[first_start:first_end].tolist()
            
            beats.append({
                'beat_number': 0,
                'peak_index': r_peaks[0],
                'leads': beat_data,
                'window_start': int(first_start),
                'window_end': int(first_end),
                'window_size': int(first_end - first_start),
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
            last_end = min((r_peaks[-1] + round((r_peaks[-1] - r_peaks[-2]) * 5.5 / 10)), signal_length)
            if last_start >= 0 and last_start < signal_length:
                beat_data = {}
                for lead_name, signal in leads.items():
                    beat_data[lead_name] = signal[last_start:last_end].tolist()
                
                beats.append({
                    'beat_number': len(r_peaks) - 1,
                    'peak_index': r_peaks[-1],
                    'leads': beat_data,
                    'window_start': int(last_start),
                    'window_end': int(last_end),
                    'window_size': int(last_end - last_start),
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
            elif self.method == 'hamilton_tompkins':
                r_peaks = self.detect_r_peaks_hamilton_tompkins(ecg_signal, fs)
            elif self.method == 'wavelet':
                r_peaks = self.detect_r_peaks_wavelet(ecg_signal, fs)
            elif self.method == 'christov':
                r_peaks = self.detect_r_peaks_christov(ecg_signal, fs)
            elif self.method == 'ensemble':
                r_peaks = self.detect_r_peaks_ensemble(ecg_signal, fs)
            elif self.method == 'pan_tompkins_simple':
                r_peaks = self.detect_r_peaks_pan_tompkins_simple(ecg_signal, fs)
            elif self.method == 'simple_enhanced':
                r_peaks = self.detect_r_peaks_simple_enhanced(ecg_signal, fs)
            else:
                raise ValueError(f"Unknown method: {self.method}")

        if len(r_peaks) == 0:
            print(f"Warning: No R-peaks detected in {filepath}")
            return []
        
        # Apply R-peak refinement if requested
        if self.refine_factor is not None:
            if args.verbose if 'args' in locals() else False:
                print(f"  Refining R-peaks with factor {self.refine_factor}")
            
            if self.min_lead_agreement > 1:
                # For multi-lead consensus, refine using the detection lead
                detection_signal = leads[detection_lead]
                r_peaks = self.refine_r_peaks(detection_signal, r_peaks, fs)
            else:
                # For single-lead detection, use the same signal
                r_peaks = self.refine_r_peaks(ecg_signal, r_peaks, fs)

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
                'refine_factor': self.refine_factor,
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
            filename = f"{base_name}_beat_{beat_num:04d}.json"
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
  
  # Use adaptive window with Pan-Tompkins detection and R-peak refinement
  python enhanced_ecg_extractor.py input.json -o output_dir --extraction adaptive -m pan_tompkins --refine-factor 0.1
  
  # Process directory with fixed windows and R-peak refinement
  python enhanced_ecg_extractor.py input_dir/ -o output_dir --extraction fixed -w 500 --refine-factor 0.15
        """
    )
    
    parser.add_argument('input', help='Input JSON file or directory')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory for extracted beats')
    parser.add_argument('-m', '--method', 
                        choices=['simple', 'pan_tompkins', 'hamilton_tompkins', 'wavelet', 'christov', 'ensemble', 'pan_tompkins_simple', 'simple_enhanced'],
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
    parser.add_argument('--refine-factor', type=float, default=None,
                        help='Interval factor (e.g., 0.1) to refine R-peak locations by searching for max values in surrounding intervals')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = EnhancedECGBeatExtractor(
        method=args.method,
        extraction_method=args.extraction,
        window_size=args.window_size,
        min_lead_agreement=args.consensus,
        tolerance_ms=args.tolerance,
        refine_factor=args.refine_factor
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
