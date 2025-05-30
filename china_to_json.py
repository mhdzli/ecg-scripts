import os
import json
import numpy as np

# === Configuration ===
file_group = "chinadat"
input_file = "input.dat"
output_file = f"{file_group}_{os.path.splitext(input_file)[0]}.json"
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
        print("Attempting to load as text file...")
        data = np.loadtxt(input_file)
        print("Successfully loaded as text file")
    except:
        # If the file is binary, read as int16
        print("Text loading failed, loading as binary int16...")
        data = np.fromfile(input_file, dtype=np.int16)
        print("Successfully loaded as binary file")
    
    print(f"Raw data length: {len(data)}")
    
    # Determine number of samples per lead (since leads are stored sequentially)
    num_samples_per_lead = len(data) // num_leads
    print(f"Samples per lead: {num_samples_per_lead}")
    print(f"Duration: {num_samples_per_lead / fs:.2f} seconds")
    
    # Reshape into (samples, leads) - stacking leads one after the other
    data_array = np.array([
        data[i * num_samples_per_lead : (i + 1) * num_samples_per_lead] 
        for i in range(num_leads)
    ]).T
    
    print(f"Reshaped data shape: {data_array.shape}")
    
    # Convert ADC values to millivolts (mV)
    print("Converting ADC values to millivolts...")
    data_array_mv = (data_array * voltage_range_mv) / adc_max_value
    
    return data_array_mv, num_samples_per_lead

def main():
    # === Load and process ECG data ===
    data_array_mv, num_samples_total = load_china_ecg_dat(
        input_file, num_leads, sampling_rate
    )
    
    # === Extract and process all leads ===
    print("Processing leads...")
    leads = {}
    
    # Store measured leads (converted to mV)
    for i, lead_name in enumerate(measured_lead_names):
        leads[lead_name] = data_array_mv[:, i].tolist()
    
    # Calculate derived leads (in mV)
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
            "source_file": input_file,
            "data_format": "China DAT format"
        },
        "leads": leads
    }
    
    # === Save to JSON file ===
    print(f"Saving to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(complete_data, f, indent=2)
    
    print(f"Successfully saved complete ECG data to '{output_file}'")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    print(f"Leads included: {', '.join(leads.keys())}")
    print(f"Data range summary:")
    
    # Show data range for each lead type
    measured_ranges = []
    derived_ranges = []
    
    for lead_name, lead_data in leads.items():
        min_val = min(lead_data)
        max_val = max(lead_data)
        range_str = f"{lead_name}: [{min_val:.3f}, {max_val:.3f}] mV"
        
        if lead_name in measured_lead_names:
            measured_ranges.append(range_str)
        else:
            derived_ranges.append(range_str)
    
    print("  Measured leads:")
    for range_info in measured_ranges:
        print(f"    {range_info}")
    
    print("  Derived leads:")
    for range_info in derived_ranges:
        print(f"    {range_info}")

if __name__ == "__main__":
    main()
