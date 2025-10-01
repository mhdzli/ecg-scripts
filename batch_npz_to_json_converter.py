import os
import numpy as np
import json
from pathlib import Path

def process_npz_files(input_dir, output_dir):
    """
    Process all NPZ files in a directory and convert each cluster to separate JSON files.
    
    Args:
        input_dir: Directory containing NPZ files
        output_dir: Directory to save JSON files
    """
    
    # Constants from original script
    STANDARD_LEADS = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all NPZ files in input directory
    npz_files = list(Path(input_dir).glob("*.npz"))
    if not npz_files:
        print(f"No NPZ files found in {input_dir}")
        return
    
    print(f"Found {len(npz_files)} NPZ files to process...")
    
    for npz_file in npz_files:
        print(f"\nProcessing {npz_file.name}...")
        
        try:
            # Load NPZ data
            d = np.load(npz_file, allow_pickle=True)
            medoids = d["medoids"]                 # [K, T_max, 12] (NaN padded)
            lengths = d["lengths"]                 # [K]
            labels  = d["cluster_labels"]          # ['0', '3_1_-1', ...]
            mask    = d["present_mask"]            # [K, 12] booleans
            lead_order = d["lead_order"].tolist() if "lead_order" in d else STANDARD_LEADS
            fs = int(d["fs"]) if "fs" in d else 1000
            patient = d["patient"].item() if "patient" in d else "Unknown"
            
            # Get base filename without extension
            base_filename = npz_file.stem
            
            print(f"  Patient: {patient}")
            print(f"  Sampling rate: {fs} Hz")
            print(f"  Number of clusters: {medoids.shape[0]}")
            
            # Process each cluster
            for i in range(medoids.shape[0]):
                arr = medoids[i]                   # [T_max, 12]
                T = int(lengths[i])
                arr = arr[:T]                      # trim NaN padding
                present = mask[i]                  # [12] which leads existed
                
                # Convert raw -> mV (same conversion as in original script)
                voltage_range_mv = 10
                adc_max_value = 2**16
                mV = (arr * voltage_range_mv) / adc_max_value
                
                # Create leads dictionary - only include present leads
                leads_data = {}
                for ch, lead in enumerate(lead_order):
                    if ch >= mV.shape[1]:
                        continue
                    if not present[ch]:
                        continue
                    # Convert to list for JSON serialization, handle NaN values
                    lead_data = mV[:, ch]
                    # Remove any remaining NaN values
                    lead_data = lead_data[~np.isnan(lead_data)]
                    leads_data[lead] = (lead_data / 2.5).tolist()
                
                # Get cluster label
                cluster_label = labels[i].item() if hasattr(labels[i], "item") else str(labels[i])
                
                # Create metadata
                metadata = {
                    "original_file": npz_file.name,
                    "sampling_rate": fs,
                    "cluster_label": cluster_label,
                    "patient": patient,
                    "present_leads": [lead_order[ch] for ch in range(len(lead_order)) 
                                    if ch < len(present) and present[ch]]
                }
                
                # Create final JSON structure
                json_data = {
                    "leads": leads_data,
                    "metadata": metadata
                }
                
                # Create output filename
                output_filename = f"{base_filename}_cluster_{i:02d}.json"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save JSON file
                with open(output_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                print(f"    Saved cluster {i} (label: {cluster_label}) -> {output_filename}")
                print(f"      Leads: {list(leads_data.keys())}")
                print(f"      Duration: {T} samples ({T/fs:.3f}s)")
                
        except Exception as e:
            print(f"  Error processing {npz_file.name}: {str(e)}")
            continue
    
    print(f"\nProcessing complete! JSON files saved to: {output_dir}")

def main():
    """
    Main function with configurable parameters
    """
    # Configuration - modify these as needed
    input_directory = "Patient_Medoid_NPZ_batch_1"  # Directory containing NPZ files
    output_directory = "json_2_output"  # Directory to save JSON files
    
    # Optional parameters that will be included in metadata
    print("NPZ to JSON Converter for ECG Data")
    print("=" * 40)
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print()
    
    process_npz_files(
        input_directory, 
        output_directory,
    )

if __name__ == "__main__":
    main()
