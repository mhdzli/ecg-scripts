import pyedflib
import json
import os
from pathlib import Path

# === Configuration ===
input_directory = "edf"  # Change this to your input directory path
output_directory = "output"  # Directory where all JSON files will be saved
segment_start = 0.0  # Start time in seconds (set your default)
segment_end = 300.0   # End time in seconds (set your default)

# === ADC conversion parameters (assume 16-bit, ±5mV range) ===
bit_depth = 16
adc_max_value = 2 ** bit_depth
voltage_range_mv = 10  # ±5mV = 10 mV total
adc_resolution = voltage_range_mv / adc_max_value * 2.5

measured_leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
derived_leads = ['III', 'aVR', 'aVL', 'aVF']
all_lead_names = measured_leads + derived_leads

def create_output_filename(file_path, input_dir, segment_start, segment_end):
    """Create output filename with subdirectory path included"""
    # Get relative path from input directory
    rel_path = os.path.relpath(file_path, input_dir)
    
    # Get directory path and filename separately
    dir_path, filename = os.path.split(rel_path)
    
    # Remove only the .edf extension (case insensitive)
    if filename.lower().endswith('.edf'):
        base_name = filename[:-4]  # Remove last 4 characters (.edf)
    else:
        base_name = os.path.splitext(filename)[0]  # Fallback to splitext
    
    # Replace path separators with underscores and create output name
    if dir_path and dir_path != '.':
        # Replace both forward and backward slashes with underscores
        dir_str = dir_path.replace('/', '_').replace('\\', '_')
        output_name = f"EDF_{dir_str}_{base_name}_SS_{int(segment_start)}_TT_{int(segment_end)}.json"
    else:
        output_name = f"EDF_{base_name}_SS_{int(segment_start)}_TT_{int(segment_end)}.json"
    
    return output_name

def process_edf_file(file_path, output_dir, segment_start, segment_end, input_dir):
    """Process a single EDF file"""
    try:
        print(f"Processing: {file_path}")
        
        # === Open EDF ===
        edf = pyedflib.EdfReader(file_path)
        n_signals = edf.signals_in_file
        labels = edf.getSignalLabels()
        sampling_rates = [edf.getSampleFrequency(i) for i in range(n_signals)]
        samples_counts = edf.getNSamples()
        
        # === Duration check ===
        max_duration = max(ns / sr for ns, sr in zip(samples_counts, sampling_rates))
        print(f"  📊 Total duration: {max_duration:.2f} seconds")
        
        if segment_end > max_duration or segment_start >= segment_end:
            print(f"  ❌ Invalid segment range for {file_path}. Skipping...")
            edf._close()
            return False
        
        # === Find ECG channel (we assume it's the first or named "ECG") ===
        ecg_index = None
        for i, label in enumerate(labels):
            if label.upper() in ["ECG", "I"]:
                ecg_index = i
                break
        
        if ecg_index is None:
            print(f"  ❌ ECG lead (I or ECG) not found in {file_path}. Skipping...")
            edf._close()
            return False
        
        sampling_rate = sampling_rates[ecg_index]
        start_sample = int(segment_start * sampling_rate)
        end_sample = int(segment_end * sampling_rate)
        total_samples = end_sample - start_sample
        
        raw_segment = edf.readSignal(ecg_index)[start_sample:end_sample]
        edf._close()
        
        # Convert ADC to mV
        lead_I_data = [val * adc_resolution for val in raw_segment]
        
        # === Build output dictionary ===
        complete_data = {
            "metadata": {
                "sampling_rate": sampling_rate,
                "total_samples": total_samples,
                "duration_seconds": total_samples / sampling_rate,
                "num_leads": 1,
                "lead_names": ["I"],
                "measured_leads": ["I"],
                "derived_leads": [],
                "units": "millivolts (mV)",
                "conversion_params": {
                    "bit_depth": bit_depth,
                    "adc_max_value": adc_max_value,
                    "voltage_range_mv": voltage_range_mv,
                    "adc_resolution": adc_resolution,
                    "conversion_formula": "raw_value * adc_resolution"
                },
                "source_file": file_path,
                "data_format": "EDF format (extracted segment)"
            },
            "leads": {
                "I": lead_I_data
            }
        }
        
        # === Save file ===
        output_filename = create_output_filename(file_path, input_dir, segment_start, segment_end)
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, "w") as f:
            json.dump(complete_data, f, indent=2)
        
        print(f"  ✅ Saved to {output_filename}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {file_path}: {str(e)}")
        return False

def main():
    global segment_start, segment_end
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Get user input for segment times (optional - you can set defaults above)
    use_defaults = input(f"Use default segment times ({segment_start}s to {segment_end}s)? (y/n): ").lower().strip()
    
    if use_defaults != 'y':
        segment_start = float(input("Enter segment start time (seconds): "))
        segment_end = float(input("Enter segment end time (seconds): "))
    
    # Find all EDF files recursively
    edf_files = []
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith('.edf'):
                edf_files.append(os.path.join(root, file))
    
    if not edf_files:
        print("❌ No EDF files found in the specified directory.")
        return
    
    print(f"📁 Found {len(edf_files)} EDF files to process...")
    print(f"📤 Output directory: {output_directory}")
    print(f"⏱️  Segment: {segment_start}s to {segment_end}s")
    print("-" * 50)
    
    # Process each file
    successful = 0
    failed = 0
    
    for file_path in edf_files:
        if process_edf_file(file_path, output_directory, segment_start, segment_end, input_directory):
            successful += 1
        else:
            failed += 1
    
    print("-" * 50)
    print(f"🎉 Processing complete!")
    print(f"✅ Successfully processed: {successful} files")
    print(f"❌ Failed: {failed} files")
    print(f"📁 All outputs saved in: {output_directory}")

if __name__ == "__main__":
    main()
