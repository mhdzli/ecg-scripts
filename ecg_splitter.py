import json
import os
import argparse
import matplotlib.pyplot as plt

def plot_ecg_segment(json_path, output_path=None):
    """
    Plot ECG data from JSON file and save as PNG.
    
    Args:
        json_path (str): Path to the JSON file with ECG data
        output_path (str): Path where to save the PNG file. If None, uses json filename with .png extension
    """
    # === Load the JSON file ===
    with open(json_path, 'r') as f:
        ecg_data = json.load(f)

    leads = ecg_data.get("leads", {})
    metadata = ecg_data.get("metadata", {})

    if not leads:
        print("No leads to plot.")
        return

    sampling_rate = metadata.get("sampling_rate", 1000)  # default to 1000 Hz
    lead_names = metadata.get("lead_names", sorted(leads.keys()))

    print(f"Plotting {len(lead_names)} leads from {os.path.basename(json_path)}...")

    num_leads = len(lead_names)
    fig, axes = plt.subplots(num_leads, 1, figsize=(12, 2 * num_leads), sharex=True)

    if num_leads == 1:
        axes = [axes]  # make iterable

    for i, lead in enumerate(lead_names):
        signal = leads[lead]
        time = [j / sampling_rate for j in range(len(signal))]
        axes[i].plot(time, signal, linewidth=0.8)
        axes[i].set_ylabel(f"{lead}\n(mV)")
        axes[i].grid(True)
        axes[i].set_xlim([0, time[-1]])

    axes[-1].set_xlabel("Time (s)")
    
    # Create title with duration info
    duration = metadata.get("duration_seconds", len(leads[lead_names[0]]) / sampling_rate)
    fig.suptitle(f"ECG Plot - {os.path.basename(json_path)} ({duration:.2f}s)", fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Determine output path
    if output_path is None:
        output_path = os.path.splitext(json_path)[0] + ".png"
    
    plt.savefig(output_path, dpi=300)
    plt.close()  # Close the figure to free memory
    print(f"✅ Plot saved as {output_path}")

def split_ecg_json(input_file, output_folder, split_time, scale_factor=None, plot_first=False):
    """
    Split an ECG JSON file into two parts: beginning segment and remaining segment.
    
    Args:
        input_file (str): Path to the complete ECG JSON file
        output_folder (str): Output folder for split files
        split_time (float): Time in seconds where to split the file
        scale_factor (float): Optional additional scale factor to apply
        plot_first (bool): Whether to generate and save a plot of the first segment
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
    total_duration = total_samples / sampling_rate
    
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Total samples: {total_samples}")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Available leads: {', '.join(leads.keys())}")
    print(f"Split time: {split_time} seconds")
    
    # === Validate split time ===
    if split_time <= 0:
        print("Error: Split time must be positive!")
        return
    if split_time >= total_duration:
        print(f"Error: Split time ({split_time}s) must be less than total duration ({total_duration:.2f}s)!")
        return
    
    # === Calculate split parameters ===
    split_sample = int(sampling_rate * split_time)
    
    first_duration = split_time
    second_duration = total_duration - split_time
    first_samples = split_sample
    second_samples = total_samples - split_sample
    
    print(f"First segment: 0 to {first_duration:.2f}s ({first_samples} samples)")
    print(f"Second segment: {first_duration:.2f}s to {total_duration:.2f}s ({second_samples} samples)")
    
    # === Create output folder ===
    os.makedirs(output_folder, exist_ok=True)
    
    # === Get base filename ===
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # === Create first segment (beginning part) ===
    print("\nProcessing first segment...")
    first_leads = {}
    for lead_name, lead_data in leads.items():
        segment_data = lead_data[:split_sample]
        
        # Apply additional scale factor if provided
        if scale_factor is not None:
            segment_data = [value * scale_factor for value in segment_data]
        
        first_leads[lead_name] = segment_data
    
    # Create first segment metadata
    first_metadata = {
        "sampling_rate": sampling_rate,
        "total_samples": first_samples,
        "duration_seconds": first_duration,
        "start_time_seconds": 0.0,
        "end_time_seconds": first_duration,
        "lead_names": list(first_leads.keys()),
        "original_scale_factor": metadata.get("scale_factor_applied", "N/A"),
        "additional_scale_factor": scale_factor if scale_factor is not None else "None",
        "split_info": {
            "is_split_file": True,
            "segment_type": "beginning",
            "original_file": os.path.basename(input_file),
            "split_time_seconds": split_time
        }
    }
    
    # Save first segment
    first_dict = {
        "metadata": first_metadata,
        "leads": first_leads
    }
    first_filename = f"{output_folder}/{base_name}_beginning_{split_time:.2f}s.json"
    with open(first_filename, "w") as f:
        json.dump(first_dict, f, indent=2)
    print(f"Saved: {first_filename}")
    
    # === Plot first segment if requested ===
    if plot_first:
        plot_filename = os.path.splitext(first_filename)[0] + ".png"
        plot_ecg_segment(first_filename, plot_filename)
    
    # === Create second segment (remaining part) ===
    print("Processing second segment...")
    second_leads = {}
    for lead_name, lead_data in leads.items():
        segment_data = lead_data[split_sample:]
        
        # Apply additional scale factor if provided
        if scale_factor is not None:
            segment_data = [value * scale_factor for value in segment_data]
        
        second_leads[lead_name] = segment_data
    
    # Create second segment metadata
    second_metadata = {
        "sampling_rate": sampling_rate,
        "total_samples": second_samples,
        "duration_seconds": second_duration,
        "start_time_seconds": first_duration,
        "end_time_seconds": total_duration,
        "lead_names": list(second_leads.keys()),
        "original_scale_factor": metadata.get("scale_factor_applied", "N/A"),
        "additional_scale_factor": scale_factor if scale_factor is not None else "None",
        "split_info": {
            "is_split_file": True,
            "segment_type": "remaining",
            "original_file": os.path.basename(input_file),
            "split_time_seconds": split_time
        }
    }
    
    # Save second segment
    second_dict = {
        "metadata": second_metadata,
        "leads": second_leads
    }
    second_filename = f"{output_folder}/{base_name}_remaining_{split_time:.2f}s.json"
    with open(second_filename, "w") as f:
        json.dump(second_dict, f, indent=2)
    print(f"Saved: {second_filename}")
    
    # === Summary ===
    first_size_mb = os.path.getsize(first_filename) / (1024*1024)
    second_size_mb = os.path.getsize(second_filename) / (1024*1024)
    total_size_mb = first_size_mb + second_size_mb
    
    print(f"\n=== Summary ===")
    print(f"First segment size: {first_size_mb:.2f} MB")
    print(f"Second segment size: {second_size_mb:.2f} MB")
    print(f"Total output size: {total_size_mb:.2f} MB")
    print(f"Successfully split ECG data at {split_time}s into '{output_folder}/'")

def main():
    parser = argparse.ArgumentParser(description="Split ECG JSON data into two files at specified time")
    parser.add_argument("input_file", help="Input JSON file with complete ECG data")
    parser.add_argument("split_time", type=float, help="Time in seconds where to split the file")
    parser.add_argument("-o", "--output", default="ecg_split", 
                       help="Output folder for split files (default: ecg_split)")
    parser.add_argument("-s", "--scale", type=float, default=None,
                       help="Additional scale factor to apply to data")
    parser.add_argument("-p", "--plot", action="store_true",
                       help="Generate plot of the first segment and save as PNG")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found!")
        return
    
    print("=== ECG JSON Splitter ===")
    print(f"Input file: {args.input_file}")
    print(f"Output folder: {args.output}")
    print(f"Split time: {args.split_time} seconds")
    if args.scale is not None:
        print(f"Scale factor: {args.scale}")
    if args.plot:
        print("Plot first segment: enabled")
    print()
    
    split_ecg_json(args.input_file, args.output, args.split_time, args.scale, args.plot)

if __name__ == "__main__":
    main()