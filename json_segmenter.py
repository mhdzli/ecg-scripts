import json
import os
import argparse

def segment_ecg_json(input_file, output_folder, segment_duration=10, scale_factor=None):
    """
    Segment a complete ECG JSON file into smaller segments.
    
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
    
    # === Extract metadata and leads ===
    metadata = complete_data["metadata"]
    leads = complete_data["leads"]
    sampling_rate = metadata["sampling_rate"]
    total_samples = metadata["total_samples"]
    
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Total samples: {total_samples}")
    print(f"Total duration: {total_samples / sampling_rate:.2f} seconds")
    print(f"Available leads: {', '.join(leads.keys())}")
    
    # === Calculate segment parameters ===
    samples_per_segment = sampling_rate * segment_duration
    num_segments = total_samples // samples_per_segment
    
    print(f"Segment duration: {segment_duration} seconds")
    print(f"Samples per segment: {samples_per_segment}")
    print(f"Number of segments: {num_segments}")
    
    # === Create output folder ===
    os.makedirs(output_folder, exist_ok=True)
    
    # === Process each segment ===
    for seg_index in range(num_segments):
        start = seg_index * samples_per_segment
        end = start + samples_per_segment
        
        # Extract segment data for each lead
        segment_leads = {}
        for lead_name, lead_data in leads.items():
            segment_data = lead_data[start:end]
            
            # Apply additional scale factor if provided
            if scale_factor is not None:
                segment_data = [value * scale_factor for value in segment_data]
            
            segment_leads[lead_name] = segment_data
        
        # Create segment metadata
        segment_metadata = {
            "sampling_rate": sampling_rate,
            "segment_index": seg_index,
            "segment_duration": segment_duration,
            "samples_per_segment": samples_per_segment,
            "start_time_seconds": seg_index * segment_duration,
            "end_time_seconds": (seg_index + 1) * segment_duration,
            "lead_names": list(segment_leads.keys()),
            "original_scale_factor": metadata.get("scale_factor_applied", "N/A"),
            "additional_scale_factor": scale_factor if scale_factor is not None else "None"
        }
        
        # Create segment dictionary
        segment_dict = {
            "metadata": segment_metadata,
            "leads": segment_leads
        }
        prefix = os.path.splitext(input_file)[0]
        # Save segment
        filename = f"{output_folder}/{prefix}_ecg_segment_{seg_index:03d}.json"
        with open(filename, "w") as f:
            json.dump(segment_dict, f, indent=2)
        
        if (seg_index + 1) % 10 == 0 or seg_index == num_segments - 1:
            print(f"Processed {seg_index + 1}/{num_segments} segments...")
    
    print(f"\nSuccessfully saved {num_segments} ECG segments to '{output_folder}/'")
    
    # === Summary ===
    total_size_mb = sum(os.path.getsize(os.path.join(output_folder, f)) 
                       for f in os.listdir(output_folder) 
                       if f.endswith('.json')) / (1024*1024)
    
    print(f"Total output size: {total_size_mb:.2f} MB")
    print(f"Average segment size: {total_size_mb/num_segments:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description="Segment ECG JSON data into smaller files")
    parser.add_argument("input_file", help="Input JSON file with complete ECG data")
    parser.add_argument("-o", "--output", default="ecg_segments", 
                       help="Output folder for segments (default: ecg_segments)")
    parser.add_argument("-d", "--duration", type=int, default=10,
                       help="Segment duration in seconds (default: 10)")
    parser.add_argument("-s", "--scale", type=float, default=None,
                       help="Additional scale factor to apply to data")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return
    
    print("=== ECG JSON Segmenter ===")
    print(f"Input file: {args.input_file}")
    print(f"Output folder: {args.output}")
    print(f"Segment duration: {args.duration} seconds")
    if args.scale is not None:
        print(f"Scale factor: {args.scale}")
    print()
    
    segment_ecg_json(args.input_file, args.output, args.duration, args.scale)

if __name__ == "__main__":
    main()
