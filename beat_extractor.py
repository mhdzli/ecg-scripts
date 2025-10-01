#!/usr/bin/env python3
"""
ECG Beat Extractor CLI Tool
Extracts individual heartbeats from custom JSON ECG files
"""

import argparse
import json
import numpy as np
import os
import sys
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
from wfdb import processing  # NEW for wfdb_xqrs

class ECGBeatExtractor:
    def __init__(self, method='simple', adaptive=False, window_size=600):
        self.method = method
        self.adaptive = adaptive
        self.window_size = window_size
        
    def load_ecg_json(self, filepath):
        """Load ECG data from custom JSON format"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', None)
            leads = data.get('leads', data)
            
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
            return filtfilt(b, a, signal)
        except:
            print("Warning: Could not apply bandpass filter, using original signal")
            return signal
    
    def detect_r_peaks_simple(self, ecg_signal, fs):
        min_distance = int(0.3 * fs)
        prominence = np.std(ecg_signal) * 0.3
        height = np.mean(ecg_signal) + prominence
        peaks, _ = find_peaks(ecg_signal, height=height, distance=min_distance, prominence=prominence)
        return peaks
    
    def detect_r_peaks_pan_tompkins(self, ecg_signal, fs):
        filtered = self.preprocess_ecg(ecg_signal, fs, 5, 15)
        derivative = np.gradient(filtered)
        squared = derivative ** 2
        window_size = int(0.150 * fs)
        window = np.ones(window_size) / window_size
        integrated = np.convolve(squared, window, mode='same')
        min_distance = int(0.2 * fs)
        threshold = np.mean(integrated) + 2 * np.std(integrated)
        peaks, _ = find_peaks(integrated, height=threshold, distance=min_distance)
        refined_peaks = []
        search_window = int(0.05 * fs)
        for peak in peaks:
            start = max(0, peak - search_window)
            end = min(len(ecg_signal), peak + search_window)
            local_max = np.argmax(ecg_signal[start:end]) + start
            refined_peaks.append(local_max)
        return np.array(refined_peaks)

    def detect_r_peaks_wfdb_xqrs(self, ecg_signal, fs):
        """R-peak detection using wfdb's XQRS"""
        xqrs = processing.XQRS(sig=ecg_signal, fs=fs)
        xqrs.detect(verbose=False)
        return np.array(xqrs.qrs_inds)

    def _split_beats_from_indices(self, leads, qrs_inds):
        """Split beats using RR midpoints across all leads"""
        beats = []
        lead_names = list(leads.keys())
        num_samples = len(leads[lead_names[0]])

        if len(qrs_inds) < 2:
            return beats

        # First segment
        beats.append({ln: leads[ln][0:(qrs_inds[0] + round((qrs_inds[1] - qrs_inds[0]) / 2))] for ln in lead_names})

        # Middle segments
        for i in range(1, len(qrs_inds) - 1):
            start = qrs_inds[i] - round((qrs_inds[i] - qrs_inds[i - 1]) * 0.5)
            end = qrs_inds[i] + round((qrs_inds[i + 1] - qrs_inds[i]) * 0.5)
            beats.append({ln: leads[ln][start:end] for ln in lead_names})

        return beats[1:]  # Skip first incomplete beat
    
    def extract_beats_from_file(self, filepath, lead='II'):
        metadata, leads = self.load_ecg_json(filepath)
        if metadata is None:
            return None
        
        if metadata is None or 'sampling_rate' not in metadata:
            fs = 500
            detection_lead = lead
            print(f"Warning: No metadata in {filepath}, assuming fs=500Hz")
        else:
            fs = metadata['sampling_rate']
            detection_lead = metadata.get('detection_lead', lead)
        
        if detection_lead not in leads:
            available_leads = list(leads.keys())
            if available_leads:
                detection_lead = available_leads[0]
                print(f"Lead {lead} not found, using {detection_lead}")
            else:
                return None
        
        ecg_signal = leads[detection_lead]
        filtered_signal = self.preprocess_ecg(ecg_signal, fs)

        if self.method == 'simple':
            r_peaks = self.detect_r_peaks_simple(filtered_signal, fs)
            if self.adaptive:
                beats = self.extract_beats_adaptive_window(ecg_signal, r_peaks, fs)
            else:
                beats = self.extract_beats_fixed_window(ecg_signal, r_peaks, self.window_size)
        elif self.method == 'pan_tompkins':
            r_peaks = self.detect_r_peaks_pan_tompkins(ecg_signal, fs)
            if self.adaptive:
                beats = self.extract_beats_adaptive_window(ecg_signal, r_peaks, fs)
            else:
                beats = self.extract_beats_fixed_window(ecg_signal, r_peaks, self.window_size)
        elif self.method == 'wfdb_xqrs':
            r_peaks = self.detect_r_peaks_wfdb_xqrs(ecg_signal, fs)
            split_leads = self._split_beats_from_indices(leads, r_peaks)
            beats = []
            for i, seg in enumerate(split_leads):
                beats.append({
                    'beat_number': i,
                    'peak_index': int(r_peaks[min(i+1, len(r_peaks)-1)]),
                    'window_size': {ln: len(seg[ln]) for ln in seg},
                    'metadata': {
                        'original_file': str(filepath),
                        'sampling_rate': fs,
                        'detection_lead': detection_lead,
                        'method': self.method,
                        'adaptive': False,
                        'total_beats': len(split_leads)
                    },
                    'leads': {ln: seg[ln].tolist() for ln in seg}
                })
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return beats

    # Keep your existing fixed/adaptive extractors here
    def extract_beats_fixed_window(self, ecg_signal, r_peaks, window_size):
        beats = []
        half_window = window_size // 2
        for i, peak in enumerate(r_peaks):
            start = max(0, peak - window_size * 4.5 // 10)
            end = min(len(ecg_signal), peak + window_size * 5.5 // 10)
            if end - start == window_size:
                beat = ecg_signal[start:end]
                beats.append({
                    'beat_number': i,
                    'peak_index': peak,
                    'signal': beat.tolist(),
                    'window_start': int(start),
                    'window_end': int(end)
                })
        return beats
    
    def extract_beats_adaptive_window(self, ecg_signal, r_peaks, fs):
        beats = []
        for i, peak in enumerate(r_peaks):
            if i == 0 and len(r_peaks) > 1:
                rr_interval = r_peaks[1] - r_peaks[0]
            elif i == len(r_peaks) - 1 and len(r_peaks) > 1:
                rr_interval = r_peaks[i] - r_peaks[i-1]
            elif len(r_peaks) > 2:
                prev_interval = r_peaks[i] - r_peaks[i-1]
                next_interval = r_peaks[i+1] - r_peaks[i]
                rr_interval = (prev_interval + next_interval) / 2
            else:
                rr_interval = int(0.8 * fs)
            window_size = int(0.6 * rr_interval)
            window_size = max(int(0.2 * fs), min(int(1.2 * fs), window_size))
            half_window = window_size // 2
            start = max(0, peak - half_window)
            end = min(len(ecg_signal), peak + half_window)
            beat = ecg_signal[start:end]
            beats.append({
                'beat_number': i,
                'peak_index': peak,
                'signal': beat.tolist(),
                'window_start': int(start),
                'window_end': int(end),
                'window_size': len(beat),
                'rr_interval_samples': int(rr_interval)
            })
        return beats

    def save_beats(self, beats, output_dir, base_filename):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        base_name = Path(base_filename).stem
        saved_files = []
        for beat in beats:
            beat_num = beat['beat_number']
            filename = f"{base_name}_{beat_num:04d}.json"
            filepath = output_path / filename
            beat_copy = self._convert_numpy_types(beat)
            with open(filepath, 'w') as f:
                json.dump(beat_copy, f, indent=2)
            saved_files.append(str(filepath))
        return saved_files
    
    def _convert_numpy_types(self, obj):
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
    parser = argparse.ArgumentParser(description='Extract heartbeats from custom JSON ECG files')
    parser.add_argument('input', help='Input JSON file or directory')
    parser.add_argument('-o', '--output', required=True, help='Output directory for extracted beats')
    parser.add_argument('-m', '--method', choices=['simple', 'pan_tompkins', 'wfdb_xqrs'],
                        default='simple', help='Peak detection method')
    parser.add_argument('-w', '--window-size', type=int, default=600,
                        help='Fixed window size in samples (ignored if --adaptive)')
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive window sizing based on RR intervals')
    parser.add_argument('-l', '--lead', default='II',
                        help='Lead to use for R-peak detection')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    
    extractor = ECGBeatExtractor(method=args.method, adaptive=args.adaptive, window_size=args.window_size)
    input_path = Path(args.input)
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
    for file_path in files_to_process:
        if args.verbose:
            print(f"Processing {file_path}...")
        beats = extractor.extract_beats_from_file(file_path, args.lead)
        if beats is None or len(beats) == 0:
            print(f"No beats found in {file_path}")
            continue
        saved_files = extractor.save_beats(beats, args.output, file_path.name)
        total_beats += len(beats)
        print(f"Extracted {len(beats)} beats from {file_path.name}")
    print(f"\nProcessed {len(files_to_process)} files, total beats: {total_beats}")

if __name__ == '__main__':
    main()

