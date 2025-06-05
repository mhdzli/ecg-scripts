import os
import json
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

# === Configuration ===
num_leads = 8
measured_lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
sampling_rate = 500  # Hz

# ECG Signal Conversion Parameters
bit_depth = 12
adc_max_value = 2**bit_depth  # 4096 for 12-bit ADC
voltage_range_mv = 10  # ±5mV = 10mV total

def load_china_ecg_dat(input_file, num_leads=8, fs=500):
    """
    Load an 8-lead ECG signal from a .DAT file and convert to millivolts (mV).
    """
    print(f"Loading ECG data from {input_file}...")
    
    try:
        # Try loading as a text file (if space/tab-separated)
        print("  Attempting to load as text file...")
        data = np.loadtxt(input_file)
        print("  Successfully loaded as text file")
    except:
        # If the file is binary, read as int16
        print("  Text loading failed, loading as binary int16...")
        data = np.fromfile(input_file, dtype=np.int16)
        print("  Successfully loaded as binary file")
    
    print(f"  Raw data length: {len(data)}")
    
    # Determine number of samples per lead (since leads are stored sequentially)
    num_samples_per_lead = len(data) // num_leads
    print(f"  Samples per lead: {num_samples_per_lead}")
    print(f"  Duration: {num_samples_per_lead / fs:.2f} seconds")
    
    # Reshape into (samples, leads) - stacking leads one after the other
    data_array = np.array([
        data[i * num_samples_per_lead : (i + 1) * num_samples_per_lead] 
        for i in range(num_leads)
    ]).T
    
    print(f"  Reshaped data shape: {data_array.shape}")
    
    # Convert ADC values to millivolts (mV)
    print("  Converting ADC values to millivolts...")
    data_array_mv = (data_array * voltage_range_mv) / adc_max_value
    
    return data_array_mv, num_samples_per_lead

def process_single_file(input_file, output_dir=None, verbose=True):
    """
    Process a single DAT file and convert to JSON.
    
    Args:
        input_file: Path to input .dat file
        output_dir: Directory to save output (default: same as input)
        verbose: Whether to print detailed progress
    
    Returns:
        tuple: (success: bool, output_file_path: str, error_message: str)
    """
    try:
        input_path = Path(input_file)
        
        # Determine output file path
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"{input_path.stem}.json"
        else:
            output_file = input_path.parent / f"{input_path.stem}.json"
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {input_file}")
            print(f"Output: {output_file}")
            print(f"{'='*60}")
        
        # === Load and process ECG data ===
        data_array_mv, num_samples_total = load_china_ecg_dat(
            input_file, num_leads, sampling_rate
        )
        
        # === Extract and process all leads ===
        if verbose:
            print("Processing leads...")
        leads = {}
        
        # Store measured leads (converted to mV)
        for i, lead_name in enumerate(measured_lead_names):
            leads[lead_name] = data_array_mv[:, i].tolist()
        
        # Calculate derived leads (in mV)
        if verbose:
            print("Calculating derived leads...")
        leads['III'] = [
            leads['II'][i] - leads['I'][i] 
            for i in range(num_samples_total)
        ]
        leads['aVR'] = [
            -(leads['I'][i] + leads['II'][i]) / 2 
            for i in range(num_samples_total)
        ]
        leads['aVL'] = [
            leads['I'][i] - leads['II'][i] / 2 
            for i in range(num_samples_total)
        ]
        leads['aVF'] = [
            leads['II'][i] - leads['I'][i] / 2 
            for i in range(num_samples_total)
        ]
        
        # === Create complete data structure ===
        complete_data = {
            "metadata": {
                "sampling_rate": sampling_rate,
                "total_samples": num_samples_total,
                "duration_seconds": num_samples_total / sampling_rate,
                "num_leads": len(leads),
                "lead_names": list(leads.keys()),
                "measured_leads": measured_lead_names,
                "derived_leads": ['III', 'aVR', 'aVL', 'aVF'],
                "units": "millivolts (mV)",
                "conversion_params": {
                    "bit_depth": bit_depth,
                    "adc_max_value": adc_max_value,
                    "voltage_range_mv": voltage_range_mv,
                    "conversion_formula": "(raw_value * voltage_range_mv) / adc_max_value"
                },
                "source_file": str(input_path.name),
                "data_format": "China DAT format",
                "processed_timestamp": datetime.now().isoformat()
            },
            "leads": leads
        }
        
        # === Save to JSON file ===
        if verbose:
            print(f"Saving to {output_file}...")
        with open(output_file, "w") as f:
            json.dump(complete_data, f, indent=2)
        
        if verbose:
            print(f"✓ Successfully saved to '{output_file}'")
            print(f"  File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
            print(f"  Leads included: {', '.join(leads.keys())}")
        
        return True, str(output_file), ""
        
    except Exception as e:
        error_msg = f"Error processing {input_file}: {str(e)}"
        print(f"✗ {error_msg}")
        return False, "", error_msg

def batch_process_dat_files(input_dir, output_dir=None, file_pattern="*.dat", verbose=True):
    """
    Process all DAT files in a directory.
    
    Args:
        input_dir: Directory containing .dat files
        output_dir: Directory to save JSON files (default: same as input_dir)
        file_pattern: Pattern to match files (default: "*.dat")
        verbose: Whether to print detailed progress
    
    Returns:
        dict: Summary of processing results
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    # Find all matching files
    dat_files = list(input_path.glob(file_pattern.lower())) + list(input_path.glob(file_pattern.upper()))
    dat_files = sorted(set(dat_files))  # Remove duplicates and sort
    
    if not dat_files:
        print(f"No files matching pattern '{file_pattern}' found in {input_dir}")
        return {"total": 0, "successful": [], "failed": []}
    
    print(f"Found {len(dat_files)} files to process")
    if verbose:
        for i, file in enumerate(dat_files, 1):
            print(f"  {i:2d}. {file.name}")
    
    # Process each file
    successful = []
    failed = []
    
    start_time = datetime.now()
    
    for i, dat_file in enumerate(dat_files, 1):
        print(f"\n[{i}/{len(dat_files)}] Processing {dat_file.name}...")
        
        success, output_file, error_msg = process_single_file(
            dat_file, output_dir, verbose=verbose
        )
        
        if success:
            successful.append({
                "input_file": str(dat_file),
                "output_file": output_file
            })
        else:
            failed.append({
                "input_file": str(dat_file),
                "error": error_msg
            })
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(dat_files)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Processing time: {duration}")
    
    if successful:
        print(f"\n✓ Successfully processed files:")
        for item in successful:
            print(f"  {Path(item['input_file']).name} → {Path(item['output_file']).name}")
    
    if failed:
        print(f"\n✗ Failed to process files:")
        for item in failed:
            print(f"  {Path(item['input_file']).name}: {item['error']}")
    
    return {
        "total": len(dat_files),
        "successful": successful,
        "failed": failed,
        "duration": str(duration)
    }

def main():
    parser = argparse.ArgumentParser(
        description="Batch convert China DAT ECG files to JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all .dat files in current directory
  python batch_china_to_json.py

  # Process all .dat files in specific directory
  python batch_china_to_json.py -i /path/to/dat/files

  # Process files and save to different output directory
  python batch_china_to_json.py -i /input/dir -o /output/dir

  # Process with custom file pattern (case-insensitive)
  python batch_china_to_json.py -i /input/dir -p "*.DAT"

  # Process single file
  python batch_china_to_json.py -f single_file.dat
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--input-dir", 
                           help="Directory containing .dat files to process")
    input_group.add_argument("-f", "--file", 
                           help="Process a single .dat file")
    
    # Output options
    parser.add_argument("-o", "--output-dir", 
                       help="Output directory for JSON files (default: same as input)")
    parser.add_argument("-p", "--pattern", default="*.dat",
                       help="File pattern to match (default: *.dat)")
    parser.add_argument("-q", "--quiet", action="store_true",
                       help="Suppress detailed output")
    
    args = parser.parse_args()
    
    try:
        if args.file:
            # Process single file
            print("Processing single file...")
            success, output_file, error_msg = process_single_file(
                args.file, args.output_dir, verbose=not args.quiet
            )
            
            if success:
                print(f"\n✓ Successfully processed: {args.file}")
                print(f"  Output: {output_file}")
            else:
                print(f"\n✗ Failed to process: {args.file}")
                print(f"  Error: {error_msg}")
                
        else:
            # Process directory
            input_dir = args.input_dir or "."
            print(f"Starting batch processing of directory: {input_dir}")
            
            results = batch_process_dat_files(
                input_dir, 
                args.output_dir, 
                args.pattern,
                verbose=not args.quiet
            )
            
            # Exit with error code if any files failed
            if results["failed"]:
                exit(1)
                
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        exit(1)

if __name__ == "__main__":
    main()