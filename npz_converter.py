import numpy as np
import json
import os
import h5py
from pathlib import Path

def load_hdf5_file(file_path):
    """Load HDF5 file and extract ECG data"""
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    if not file_path.exists():
        raise FileNotFoundError(f"No such file: {file_path}")
    
    data_dict = {}
    with h5py.File(file_path, "r") as hf:
        print(f"HDF5 keys: {list(hf.keys())}")
        
        for key in hf.keys():
            value = hf[key]
            if isinstance(value, h5py.Dataset):
                array_data = np.array(value)
                print(f"  Dataset '{key}': shape={array_data.shape}, dtype={array_data.dtype}")
                data_dict[key] = array_data
                
            elif isinstance(value, h5py.Group):
                print(f"  Group '{key}' contains: {list(value.keys())}")
                group_data = []
                for subkey in value.keys():
                    subvalue = value[subkey]
                    if isinstance(subvalue, h5py.Dataset):
                        array_data = np.array(subvalue)
                        print(f"    Dataset '{subkey}': shape={array_data.shape}, dtype={array_data.dtype}")
                        group_data.append(array_data)
                data_dict[key] = group_data
    
    print(f"Loaded HDF5 file '{file_path}' with keys: {list(data_dict.keys())}")
    return data_dict

def convert_hdf5_to_json(input_file, output_dir):
    """Convert HDF5 file with 13-lead ECG clusters to JSON format"""
    # === ECG Lead Configuration ===
    # Standard 12-lead + 1 rhythm lead (usually lead II)
    ecg_leads = [
        "I", "II", "III",           # Limb leads
        "aVR", "aVL", "aVF",        # Augmented limb leads  
        "V1", "V2", "V3", "V4", "V5", "V6",  # Precordial leads
        "II_rhythm"                 # Rhythm strip (13th lead)
    ]
    
    measured_leads = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
    derived_leads = ["III", "aVR", "aVL", "aVF", "II_rhythm"]
    
    # === Conversion config ===
    bit_depth = 16
    adc_max_value = 2 ** bit_depth
    voltage_range_mv = 10
    adc_resolution = voltage_range_mv / adc_max_value
    sampling_rate = 1000  # Hz assumed
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load HDF5 data
    data = load_hdf5_file(input_file)
    
    # Handle different possible structures in HDF5
    tensor_data = None
    if "tensor_group" in data:
        tensor_data = data["tensor_group"]
        print(f"Using tensor_group with {len(tensor_data)} items")
    else:
        # Try to find array data
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if value.ndim >= 2 and (13 in value.shape):
                    tensor_data = [value]  # Single ECG recording
                    print(f"Using single array from key: {key}")
                    break
            elif isinstance(value, list) and len(value) > 0:
                # Check if it's a list of arrays that could be ECG data
                first_item = value[0]
                if isinstance(first_item, np.ndarray) and first_item.ndim >= 1:
                    tensor_data = value
                    print(f"Using list of arrays from key: {key}")
                    break
    
    if tensor_data is None:
        print("❌ No suitable tensor data found.")
        print("Available data shapes:")
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
        return
    
    print(f"📦 Processing {len(tensor_data)} ECG recordings")
    
    # Process each ECG recording
    for idx, ecg_data in enumerate(tensor_data):
        ecg_array = np.array(ecg_data)
        print(f"ECG {idx}: shape={ecg_array.shape}")
        
        # Handle different array orientations for 13-lead ECG
        if ecg_array.ndim == 1:
            # Flat array - might be 13 leads concatenated
            if len(ecg_array) % 13 == 0:
                samples_per_lead = len(ecg_array) // 13
                ecg_array = ecg_array.reshape(13, samples_per_lead)
                print(f"  Reshaped to (13, {samples_per_lead})")
            else:
                print(f"  ⚠️ Cannot reshape 1D array of length {len(ecg_array)} into 13 leads")
                continue
                
        elif ecg_array.ndim == 2:
            if ecg_array.shape[0] == 13:
                # Shape: (13_leads, n_samples) - correct orientation
                pass
            elif ecg_array.shape[1] == 13:
                # Shape: (n_samples, 13_leads) - transpose needed
                ecg_array = ecg_array.T
                print(f"  Transposed to shape: {ecg_array.shape}")
            else:
                print(f"  ⚠️ Unexpected 2D shape: {ecg_array.shape} (expected 13 in one dimension)")
                continue
        else:
            print(f"  ⚠️ Unsupported array dimensions: {ecg_array.ndim}")
            continue
        
        if ecg_array.shape[0] != 13:
            print(f"  ⚠️ Expected 13 leads, got {ecg_array.shape[0]}")
            continue
            
        n_samples = ecg_array.shape[1]
        print(f"  Processing 13 leads with {n_samples} samples each")
        
        # Convert each lead to millivolts
        leads_data = {}
        for lead_idx, lead_name in enumerate(ecg_leads):
            signal = ecg_array[lead_idx]
            signal_mv = (signal * adc_resolution).tolist()
            leads_data[lead_name] = signal_mv
        
        # Validate ECG data
        if validate_ecg_leads(leads_data):
            print(f"  ✅ ECG validation passed")
        else:
            print(f"  ⚠️ ECG validation warnings (check above)")
        
        json_dict = {
            "metadata": {
                "sampling_rate": sampling_rate,
                "total_samples": n_samples,
                "duration_seconds": n_samples / sampling_rate,
                "num_leads": len(leads_data),
                "lead_names": list(leads_data.keys()),
                "measured_leads": [lead for lead in measured_leads if lead in leads_data],
                "derived_leads": [lead for lead in derived_leads if lead in leads_data],
                "units": "millivolts (mV)",
                "conversion_params": {
                    "bit_depth": bit_depth,
                    "adc_max_value": adc_max_value,
                    "voltage_range_mv": voltage_range_mv,
                    "adc_resolution": adc_resolution,
                    "conversion_formula": "raw_value * adc_resolution"
                },
                "source_file": str(input_file),
                "data_format": "HDF5 13-lead ECG cluster",
                "ecg_type": "12-lead + rhythm strip"
            },
            "leads": leads_data
        }
        
        output_file = os.path.join(output_dir, f"ECG_13lead_cluster_{idx:04d}.json")
        with open(output_file, "w") as f:
            json.dump(json_dict, f, indent=2)
        
        print(f"  💾 Saved {output_file}")
    
    print("\n🎉 All 13-lead ECG clusters exported to JSON format.")

def inspect_hdf5_structure(file_path):
    """Inspect HDF5 file structure to understand the ECG data layout"""
    with h5py.File(file_path, "r") as hf:
        print(f"HDF5 file: {file_path}")
        print(f"Root keys: {list(hf.keys())}")
        print("\nDetailed structure:")
        
        def print_structure(name, obj):
            indent = "  " * (name.count('/'))
            if isinstance(obj, h5py.Dataset):
                shape_info = f"shape={obj.shape}, dtype={obj.dtype}"
                
                # ECG-specific interpretation
                if len(obj.shape) == 1:
                    if obj.shape[0] % 13 == 0:
                        samples_per_lead = obj.shape[0] // 13
                        print(f"{indent}📊 {name}: {shape_info}")
                        print(f"{indent}   → Possibly 13 leads × {samples_per_lead} samples (flattened)")
                elif len(obj.shape) == 2:
                    if 13 in obj.shape:
                        print(f"{indent}📊 {name}: {shape_info}")
                        print(f"{indent}   → Likely 13-lead ECG data")
                    else:
                        print(f"{indent}📊 {name}: {shape_info}")
                else:
                    print(f"{indent}📊 {name}: {shape_info}")
                    
                # Show sample values for small arrays
                if obj.size <= 20:
                    print(f"{indent}   Sample: {obj[...]}")
                elif obj.size <= 1000:
                    print(f"{indent}   Sample: {obj.flat[:5]}... (first 5 values)")
                    
            elif isinstance(obj, h5py.Group):
                print(f"{indent}📁 {name}/ (group with {len(obj)} items)")
        
        hf.visititems(print_structure)
    
    return None

def validate_ecg_leads(leads_data, expected_leads=13):
    """Validate ECG lead data structure"""
    print(f"    📊 ECG Validation:")
    print(f"    Expected leads: {expected_leads}")
    print(f"    Found leads: {len(leads_data)}")
    
    # Check lead naming
    standard_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    found_standard = [lead for lead in standard_leads if lead in leads_data]
    print(f"    Standard 12-leads found: {len(found_standard)}/12")
    if len(found_standard) < 12:
        missing = [lead for lead in standard_leads if lead not in leads_data]
        print(f"    Missing standard leads: {missing}")
    
    # Check sample counts
    sample_counts = [len(data) for data in leads_data.values()]
    if len(set(sample_counts)) == 1:
        print(f"    ✅ All leads have consistent length: {sample_counts[0]} samples")
    else:
        print(f"    ⚠️ Inconsistent lead lengths: {set(sample_counts)}")
    
    # Check for reasonable ECG values (after conversion to mV)
    max_values = [max(abs(min(data)), abs(max(data))) for data in leads_data.values()]
    avg_max = sum(max_values) / len(max_values)
    if 0.1 <= avg_max <= 50:  # Reasonable ECG range in mV
        print(f"    ✅ ECG amplitude range looks reasonable: ~{avg_max:.2f} mV")
    else:
        print(f"    ⚠️ ECG amplitude might be unusual: ~{avg_max:.2f} mV")
    
    return len(leads_data) == expected_leads and len(set(sample_counts)) == 1

# Usage examples
if __name__ == "__main__":
    # Method 1: Inspect the HDF5 structure first (recommended)
    input_file = "cluster_5.npz"  # Actually an HDF5 file
    
    print("🔍 Inspecting HDF5 file structure...")
    try:
        inspect_hdf5_structure(input_file)
    except Exception as e:
        print(f"❌ Inspection failed: {e}")
        print("Please check if the file exists and is a valid HDF5 file")
        exit()
    
    # Method 2: Convert 13-lead ECG clusters
    output_dir = "json_ecg_13lead_output"
    
    print(f"\n🔄 Converting {input_file} to JSON...")
    try:
        convert_hdf5_to_json(input_file, output_dir)
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
    
    # Method 3: Batch process multiple HDF5 files
    # hdf5_files = ["cluster_1.npz", "cluster_2.npz", "cluster_3.npz", "cluster_4.npz", "cluster_5.npz"]
    # for hdf5_file in hdf5_files:
    #     if os.path.exists(hdf5_file):
    #         convert_hdf5_to_json(hdf5_file, f"json_output_{Path(hdf5_file).stem}")
    
    # Note: Files will be named as ECG_13lead_cluster_0000.json, ECG_13lead_cluster_0001.json, etc.
