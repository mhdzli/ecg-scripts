import os
import json
import re

# === Configuration ===
file_group = "bard"
input_file = "input_file.txt"  # Update this path
output_file = f"{file_group}_{os.path.splitext(input_file)[0]}.json"

# ECG Signal Conversion Parameters
bit_depth = 16
adc_max_value = 2**bit_depth  # 65536 for 16-bit ADC
voltage_range_mv = 10  # ±5mV = 10mV total
adc_resolution = voltage_range_mv / adc_max_value  # mV per ADC unit

# All 12 standard ECG leads (including derived leads)
all_lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
measured_leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']  # Typically measured
derived_leads = ['III', 'aVR', 'aVL', 'aVF']  # Typically derived

def parse_bard_ecg_file(input_file):
    """
    Parse Bard ECG text file and extract sampling rate and lead data.
    """
    print(f"Loading ECG data from {input_file}...")
    
    # Initialize variables
    lead_data = {lead: [] for lead in all_lead_names}
    sampling_rate = None
    data_section = False
    line_count = 0
    data_line_count = 0
    
    with open(input_file, 'r') as f:
        for line in f:
            line_count += 1
            line = line.strip()
            
            # Extract sampling rate (first occurrence)
            if not sampling_rate and 'Sample Rate:' in line:
                match = re.search(r'Sample Rate:\s*(\d+)Hz', line)
                if match:
                    sampling_rate = int(match.group(1))
                    print(f"Found sampling rate: {sampling_rate} Hz")
            
            # Check if we've reached the data section
            if line == '[Data]':
                data_section = True
                print("Found data section, starting to parse ECG data...")
                continue
            
            # Process data lines
            if data_section and line:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Parse the comma-separated values
                values = line.split(',')
                
                # Make sure we have exactly 12 values for the 12 leads
                if len(values) == 12:
                    data_line_count += 1
                    for i, lead_name in enumerate(all_lead_names):
                        try:
                            # Convert ADC value to millivolts
                            raw_value = float(values[i])
                            mv_value = raw_value * adc_resolution
                            lead_data[lead_name].append(mv_value)
                        except ValueError:
                            print(f"Warning: Invalid value '{values[i]}' found at line {line_count}")
                            # Use 0 as fallback
                            lead_data[lead_name].append(0.0)
                elif len(values) > 1:  # Ignore single values or headers
                    print(f"Warning: Expected 12 values but found {len(values)} at line {line_count}: {line[:50]}...")
    
    print(f"Processed {line_count} total lines")
    print(f"Extracted {data_line_count} data lines")
    
    # Verify all leads have the same length
    lead_lengths = {lead: len(data) for lead, data in lead_data.items()}
    unique_lengths = set(lead_lengths.values())
    
    if len(unique_lengths) > 1:
        print("Warning: Leads have different lengths!")
        for lead, length in lead_lengths.items():
            print(f"  {lead}: {length} samples")
    else:
        total_samples = list(unique_lengths)[0]
        print(f"All leads have {total_samples} samples")
    
    return lead_data, sampling_rate, total_samples

def main():
    # === Load and process ECG data ===
    leads, sampling_rate, num_samples_total = parse_bard_ecg_file(input_file)
    
    if sampling_rate is None:
        print("Warning: Could not extract sampling rate from file. Using default 500 Hz.")
        sampling_rate = 500
    
    # Calculate duration
    duration_seconds = num_samples_total / sampling_rate
    
    print(f"Total samples: {num_samples_total}")
    print(f"Duration: {duration_seconds:.2f} seconds")
    print(f"Sampling rate: {sampling_rate} Hz")
    
    # === Create complete data structure ===
    complete_data = {
        "metadata": {
            "sampling_rate": sampling_rate,
            "total_samples": num_samples_total,
            "duration_seconds": duration_seconds,
            "num_leads": len(leads),
            "lead_names": list(leads.keys()),
            "measured_leads": measured_leads,
            "derived_leads": derived_leads,
            "units": "millivolts (mV)",
            "conversion_params": {
                "bit_depth": bit_depth,
                "adc_max_value": adc_max_value,
                "voltage_range_mv": voltage_range_mv,
                "adc_resolution": adc_resolution,
                "conversion_formula": "raw_value * adc_resolution"
            },
            "source_file": input_file,
            "data_format": "Bard TXT format"
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
    
    # === Show data range summary ===
    print(f"Data range summary:")
    
    measured_ranges = []
    derived_ranges = []
    
    for lead_name, lead_data in leads.items():
        if len(lead_data) > 0:
            min_val = min(lead_data)
            max_val = max(lead_data)
            mean_val = sum(lead_data) / len(lead_data)
            range_str = f"{lead_name}: [{min_val:.3f}, {max_val:.3f}] mV (mean: {mean_val:.3f})"
            
            if lead_name in measured_leads:
                measured_ranges.append(range_str)
            else:
                derived_ranges.append(range_str)
    
    print("  Measured leads:")
    for range_info in measured_ranges:
        print(f"    {range_info}")
    
    print("  Derived leads:")
    for range_info in derived_ranges:
        print(f"    {range_info}")
    
    # === Data quality checks ===
    print("\nData quality checks:")
    
    # Check for leads with all zeros
    zero_leads = [lead for lead, data in leads.items() if all(val == 0.0 for val in data)]
    if zero_leads:
        print(f"  Warning: Leads with all zero values: {', '.join(zero_leads)}")
    else:
        print("  ✓ No leads with all zero values")
    
    # Check for extremely high values (potential parsing errors)
    high_value_leads = []
    for lead, data in leads.items():
        if any(abs(val) > 50 for val in data):  # Values > 50mV are suspicious
            max_abs = max(abs(val) for val in data)
            high_value_leads.append(f"{lead} (max: {max_abs:.3f} mV)")
    
    if high_value_leads:
        print(f"  Warning: Leads with suspiciously high values: {', '.join(high_value_leads)}")
    else:
        print("  ✓ No suspiciously high values detected")

if __name__ == "__main__":
    main()
