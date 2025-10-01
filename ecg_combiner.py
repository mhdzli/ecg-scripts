#!/usr/bin/env python3
"""
ECG Signal Combiner
Upsamples EDF ECG from 250 Hz to 1000 Hz and adds it as a new lead to Holter data
"""

import argparse
import json
import numpy as np
from scipy import signal
from pathlib import Path

def load_ecg_json(filepath):
    """Load ECG data from JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        leads = data.get('leads', data)
        
        # Convert leads to numpy arrays
        for lead_name in leads:
            if isinstance(leads[lead_name], list):
                leads[lead_name] = np.array(leads[lead_name])
        
        return metadata, leads
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None

def upsample_signal(signal_data, original_fs, target_fs):
    """
    Upsample signal from original_fs to target_fs using linear interpolation
    For 250 Hz to 1000 Hz: adds 3 points between each pair of original points
    """
    if original_fs == target_fs:
        return signal_data
    
    # Calculate upsampling ratio
    ratio = target_fs / original_fs
    
    if ratio != 4.0:
        print(f"Warning: Upsampling ratio is {ratio}, not 4. Using scipy resampling instead of simple interpolation.")
        # Fallback to scipy for non-4x ratios
        if ratio != int(ratio):
            num_samples = int(len(signal_data) * ratio)
            upsampled = signal.resample(signal_data, num_samples)
        else:
            up = int(ratio)
            upsampled = signal.resample_poly(signal_data, up, 1)
        return upsampled
    
    # Linear interpolation: add 3 points between each pair of original points
    # For 4x upsampling (250 Hz -> 1000 Hz)
    n_original = len(signal_data)
    n_upsampled = (n_original - 1) * 4 + 1  # Each gap gets 3 new points
    
    upsampled = np.zeros(n_upsampled)
    
    for i in range(n_original - 1):
        # Original points
        y0 = signal_data[i]
        y1 = signal_data[i + 1]
        
        # Calculate positions in upsampled array
        base_idx = i * 4
        
        # Place original point
        upsampled[base_idx] = y0
        
        # Add 3 interpolated points between y0 and y1
        for j in range(1, 4):
            # Linear interpolation: weight = j/4
            weight = j / 4.0
            upsampled[base_idx + j] = y0 + weight * (y1 - y0)
    
    # Add the last original point
    upsampled[-1] = signal_data[-1]
    
    return upsampled

def time_to_samples(time_seconds, sampling_rate):
    """Convert time in seconds to sample index"""
    return int(time_seconds * sampling_rate)

def combine_ecg_files(edf_file, holter_file, holter_start_time, edf_lead_name='I', 
                      new_lead_name='BG', output_file=None, use_samples=False):
    """
    Combine EDF and Holter ECG files
    
    Parameters:
    -----------
    edf_file : str
        Path to EDF JSON file (250 Hz)
    holter_file : str
        Path to Holter JSON file (1000 Hz)
    holter_start_time : float
        Time in seconds (or sample number if use_samples=True) when Holter recording started relative to EDF recording
    edf_lead_name : str
        Name of the lead in EDF file to extract (default: 'I')
    new_lead_name : str
        Name for the new lead in combined file (default: 'BG')
    output_file : str
        Path for output JSON file (if None, auto-generate name)
    use_samples : bool
        If True, holter_start_time is interpreted as sample number in original EDF (at 250 Hz)
    """
    
    print(f"Loading EDF file: {edf_file}")
    edf_metadata, edf_leads = load_ecg_json(edf_file)
    if edf_leads is None:
        return None
    
    print(f"Loading Holter file: {holter_file}")
    holter_metadata, holter_leads = load_ecg_json(holter_file)
    if holter_leads is None:
        return None
    
    # Get sampling rates
    edf_fs = edf_metadata.get('sampling_rate', 250)
    holter_fs = holter_metadata.get('sampling_rate', 1000)
    
    print(f"\nEDF sampling rate: {edf_fs} Hz")
    print(f"Holter sampling rate: {holter_fs} Hz")
    
    if use_samples:
        edf_start_sample = int(holter_start_time)
        holter_start_time_sec = edf_start_sample / edf_fs
        print(f"Holter starts at: sample {edf_start_sample} in EDF recording ({holter_start_time_sec:.2f} seconds)")
    else:
        holter_start_time_sec = holter_start_time
        edf_start_sample = time_to_samples(holter_start_time, edf_fs)
        print(f"Holter starts at: {holter_start_time} seconds in EDF recording (sample {edf_start_sample})")
    
    # Check if EDF lead exists
    if edf_lead_name not in edf_leads:
        available = list(edf_leads.keys())
        print(f"Error: Lead '{edf_lead_name}' not found in EDF file.")
        print(f"Available leads: {available}")
        if available:
            edf_lead_name = available[0]
            print(f"Using first available lead: {edf_lead_name}")
        else:
            return None
    
    # Extract EDF lead
    edf_signal = edf_leads[edf_lead_name]
    print(f"\nEDF signal length: {len(edf_signal)} samples ({len(edf_signal)/edf_fs:.2f} seconds)")
    
    # Upsample EDF signal to match Holter sampling rate
    print(f"\nUpsampling EDF from {edf_fs} Hz to {holter_fs} Hz...")
    upsampled_edf = upsample_signal(edf_signal, edf_fs, holter_fs)
    print(f"Upsampled signal length: {len(upsampled_edf)} samples ({len(upsampled_edf)/holter_fs:.2f} seconds)")
    
    # Calculate alignment in the upsampled signal
    # If using sample numbers, convert from original EDF sample to upsampled sample
    if use_samples:
        holter_start_sample = int(edf_start_sample * (holter_fs / edf_fs))
    else:
        holter_start_sample = time_to_samples(holter_start_time_sec, holter_fs)
    
    print(f"\nHolter start sample in upsampled EDF: {holter_start_sample}")
    
    # Get Holter signal length
    holter_length = len(next(iter(holter_leads.values())))
    print(f"Holter signal length: {holter_length} samples ({holter_length/holter_fs:.2f} seconds)")
    
    # Extract the overlapping portion of EDF that corresponds to Holter recording
    edf_end_sample = holter_start_sample + holter_length
    
    if holter_start_sample < 0:
        print(f"Error: Holter start time is negative (before EDF recording started)")
        return None
    
    if holter_start_sample >= len(upsampled_edf):
        print(f"Error: Holter start time is beyond EDF recording")
        return None
    
    # Extract and align the EDF segment
    if edf_end_sample <= len(upsampled_edf):
        # EDF covers entire Holter recording
        aligned_edf = upsampled_edf[holter_start_sample:edf_end_sample]
        print(f"\nEDF covers entire Holter recording")
    else:
        # EDF is shorter than Holter, pad with zeros
        available_samples = len(upsampled_edf) - holter_start_sample
        aligned_edf = np.zeros(holter_length)
        aligned_edf[:available_samples] = upsampled_edf[holter_start_sample:]
        print(f"\nWarning: EDF ends before Holter. Padding last {holter_length - available_samples} samples with zeros")
    
    print(f"Aligned EDF segment length: {len(aligned_edf)} samples")
    
    # Create combined data structure - convert all numpy arrays to lists
    combined_leads = {}
    for lead_name, lead_data in holter_leads.items():
        if isinstance(lead_data, np.ndarray):
            combined_leads[lead_name] = lead_data.tolist()
        else:
            combined_leads[lead_name] = lead_data
    
    # Add the new lead
    combined_leads[new_lead_name] = aligned_edf.tolist()
    
    # Update metadata
    combined_metadata = holter_metadata.copy()
    combined_metadata['combined_from_edf'] = str(edf_file)
    combined_metadata['edf_lead_source'] = edf_lead_name
    if use_samples:
        combined_metadata['holter_start_sample_in_edf'] = edf_start_sample
        combined_metadata['holter_start_time_in_edf'] = holter_start_time_sec
    else:
        combined_metadata['holter_start_time_in_edf'] = holter_start_time_sec
    combined_metadata['edf_original_fs'] = edf_fs
    combined_metadata['edf_upsampled_to_fs'] = holter_fs
    
    # Prepare output
    output_data = {
        'metadata': combined_metadata,
        'leads': combined_leads
    }
    
    # Determine output filename
    if output_file is None:
        holter_path = Path(holter_file)
        output_file = holter_path.parent / f"{holter_path.stem}_combined.json"
    
    # Save combined file
    print(f"\nSaving combined file: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSuccess! Combined ECG saved to: {output_file}")
    print(f"Total leads in combined file: {len(combined_leads)}")
    print(f"Lead names: {list(combined_leads.keys())}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(
        description='Combine EDF and Holter ECG files with upsampling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Holter started 30 seconds into EDF recording
  python combine_ecg.py edf.json holter.json 30.0 -o combined.json
  
  # Holter started at sample 7500 in EDF (250 Hz = 30 seconds)
  python combine_ecg.py edf.json holter.json 7500 --use-samples -o combined.json
  
  # Holter started 2 minutes into EDF, use custom lead names
  python combine_ecg.py edf.json holter.json 120.0 --edf-lead I --new-lead-name BG
  
  # Holter started at sample 30000 (250 Hz = 120 seconds)
  python combine_ecg.py edf.json holter.json 30000 --use-samples
  
  # Holter started immediately with EDF
  python combine_ecg.py edf.json holter.json 0.0
        """
    )
    
    parser.add_argument('edf_file', help='EDF JSON file (250 Hz)')
    parser.add_argument('holter_file', help='Holter JSON file (1000 Hz)')
    parser.add_argument('holter_start', type=float,
                       help='When Holter started: time in seconds (default) or sample number (with --use-samples)')
    parser.add_argument('-o', '--output', help='Output JSON file (default: auto-generated)')
    parser.add_argument('--edf-lead', default='I',
                       help='Lead name to extract from EDF file (default: I)')
    parser.add_argument('--new-lead-name', default='BG',
                       help='Name for the new lead in combined file (default: BG)')
    parser.add_argument('--use-samples', action='store_true',
                       help='Interpret holter_start as sample number in EDF (at 250 Hz) instead of seconds')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        output_file = combine_ecg_files(
            args.edf_file,
            args.holter_file,
            args.holter_start,
            edf_lead_name=args.edf_lead,
            new_lead_name=args.new_lead_name,
            output_file=args.output,
            use_samples=args.use_samples
        )
        
        if output_file:
            print("\n" + "="*60)
            print("COMBINATION COMPLETE")
            print("="*60)
            print(f"Use this file with the ECG beat extractor:")
            print(f"  python enhanced_ecg_extractor.py {output_file} -o output_dir --plot-r-peaks")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
