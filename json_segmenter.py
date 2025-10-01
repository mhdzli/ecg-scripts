import json
import os
import argparse
from pathlib import Path

def segment_ecg_json(input_file, output_folder, segment_duration=10, scale_factor=None):
    """
    Segment a complete ECG JSON file into smaller segments.
    Works with various JSON structures including combined files.
    
    Args:
        input_file (str): Path to the complete ECG JSON file
        output_folder (str): Output folder for segments
        segment_duration (int): Duration of each segment in seconds
        scale_factor (float): Optional additional scale factor to apply
    """
    
    # === Load the complete JSON file ===
    print(f"Loading ECG data from {input_file}...")
    with open(input_file, "r") as f:
        complete_data = json.load(f)
    
    # === Extract metadata and leads with flexible structure handling ===
    # Handle both {"metadata": {...}, "leads": {...}} and direct lead structure
    if "metadata" in complete_data and "leads" in complete_data:
        metadata = complete_data["metadata"]
        leads = complete_data["leads"]
    elif "metadata" in complete_data:
        # Metadata exists, but leads might be at top level
        metadata = complete_data["metadata"]
        leads = {k: v for k, v in complete_data.items() if k != "metadata" and isinstance(v, list)}
    else:
        # No metadata, assume everything is leads
        metadata = {}
        leads = {k: v for k, v in complete_data.items() if isinstance(v, list)}
    
    if not leads:
        print("Error: No lead data found in JSON file!")
        return
    
    # === Get or calculate sampling rate ===
    if "sampling_rate" in metadata:
        sampling_rate = metadata["sampling_rate"]
    else:
        print("Warning: No sampling_rate in metadata, assuming 500 Hz")
        sampling_rate = 500
    
    # === Get or calculate total samples ===
    # Check all leads to ensure they have the same length
    lead_lengths = {name: len(data) for name, data in leads.items()}
    
    if len(set(lead_lengths.values())) > 1:
        print("Warning: Leads have different lengths!")
        for name, length in lead_lengths.items():
            print(f"  {name}: {length} samples")
        print("Using the minimum length for segmentation")
        total_samples = min(lead_lengths.values())
    else:
        total_samples = list(lead_lengths.values())[0]
    
    # Verify against metadata if available
    if "total_samples" in metadata and metadata["total_samples"] != total_samples:
        print(f"Warning: metadata total_samples ({metadata['total_samples']}) differs from actual data length ({total_samples})")
        print(f"Using actual data length: {total_samples}")
    
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Total samples: {total_samples}")
    print(f"Total duration: {total_samples / sampling_rate:.2f} seconds")
    print(f"Available leads: {', '.join(leads.keys())}")
    
    # === Calculate segment parameters ===
    samples_per_segment = sampling_rate * segment_duration
    num_segments = total_samples // samples_per_segment
    remaining_samples = total_samples % samples_per_segment
    
    print(f"Segment duration: {segment_duration} seconds")
    print(f"Samples per segment: {samples_per_segment}")
    print(f"Number of complete segments: {num_segments}")
    if remaining_samples > 0:
        print(f"Remaining samples (not segmented): {remaining_samples} ({remaining_samples/sampling_rate:.2f} seconds)")
    
    # === Create output folder ===
    os.makedirs(output_folder, exist_ok=True)
    
    # Get base filename without extension
    input_path = Path(input_file)
    base_filename = input_path.stem
    
    # === Process each segment ===
    for seg_index in range(num_segments):
        start = seg_index * samples_per_segment
        end = start + samples_per_segment
        
        # Extract segment data for each lead
        segment_leads = {}
        for lead_name, lead_data in leads.items():
            # Ensure we don't go beyond the data length
            segment_data = lead_data[start:min(end, len(lead_data))]
            
            # Apply additional scale factor if provided
            if scale_factor is not None:
                segment_data = [value * scale_factor for value in segment_data]
            
            segment_leads[lead_name] = segment_data
        
        # Create segment metadata - preserve original metadata and add segment info
        segment_metadata = metadata.copy()
        segment_metadata.update({
            "sampling_rate": sampling_rate,
            "segment_index": seg_index,
            "segment_duration": segment_duration,
            "samples_per_segment": samples_per_segment,
            "total_samples": len(segment_leads[list(segment_leads.keys())[0]]),
            "start_time_seconds": seg_index * segment_duration,
            "end_time_seconds": (seg_index + 1) * segment_duration,
            "lead_names": list(segment_leads.keys()),
            "original_file": str(input_path.name),
            "additional_scale_factor": scale_factor if scale_factor is not None else "None"
        })
        
        # Preserve original scale factor if it exists
        if "scale_factor_applied" in metadata:
            segment_metadata["original_scale_factor"] = metadata["scale_factor_applied"]
        
        # Create segment dictionary
        segment_dict = {
            "metadata": segment_metadata,
            "leads": segment_leads
        }
        
        # Save segment
        filename = f"{output_folder}/{base_filename}_segment_{seg_index:03d}.json"
        with open(filename, "w") as f:
            json.dump(segment_dict, f, indent=2)
        
        if (seg_index + 1) % 10 == 0 or seg_index == num_segments - 1:
            print(f"Processed {seg_index + 1}/{num_segments} segments...")
    
    print(f"\nSuccessfully saved {num_segments} ECG segments to '{output_folder}/'")
    
    # === Summary ===
    output_files = [f for f in os.listdir(output_folder) if f.endswith('.json')]
    if output_files:
        total_size_mb = sum(os.path.getsize(os.path.join(output_folder, f)) 
                           for f in output_files) / (1024*1024)
        
        print(f"Total output size: {total_size_mb:.2f} MB")
        print(f"Average segment size: {total_size_mb/num_segments:.2f} MB")

def main():
    parser = argparse.ArgumentParser(
        description="Segment ECG JSON data into smaller files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Segment into 10-second segments
  python json_segmenter.py combined.json -o segments -d 10
  
  # Segment into 30-second segments with scale factor
  python json_segmenter.py ecg.json -o output -d 30 -s 1.5
  
  # Segment with custom output folder
  python json_segmenter.py holter.json -o holter_segments -d 60
        """
    )
    
    parser.add_argument("input_file", help="Input JSON file with complete ECG data")
    parser.add_argument("-o", "--output", default="ecg_segments", 
                       help="Output folder for segments (default: ecg_segments)")
    parser.add_argument("-d", "--duration", type=int, default=10,
                       help="Segment duration in seconds (default: 10)")
    parser.add_argument("-s", "--scale", type=float, default=None,
                       help="Additional scale factor to apply to data")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return 1
    
    print("=== ECG JSON Segmenter ===")
    print(f"Input file: {args.input_file}")
    print(f"Output folder: {args.output}")
    print(f"Segment duration: {args.duration} seconds")
    if args.scale is not None:
        print(f"Scale factor: {args.scale}")
    print()
    
    try:
        segment_ecg_json(args.input_file, args.output, args.duration, args.scale)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
