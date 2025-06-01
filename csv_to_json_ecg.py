import os
import json
import csv
import numpy as np

# === Configuration ===
file_group = "csv_ecg"
input_file = "BRS 2EP.csv"  # Update this path
output_file = f"{file_group}_{os.path.splitext(input_file)[0].replace(' ', '_')}.json"

# Default sampling rate (update if known)
default_sampling_rate = 500  # Hz - adjust if you know the actual sampling rate

# All 12 standard ECG leads (in the order they appear in your CSV)
all_lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
measured_leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']  # Typically measured
derived_leads = ['III', 'aVR', 'aVL', 'aVF']  # Typically derived

def parse_csv_ecg_file(input_file):
    """
    Parse CSV ECG file and extract lead data.
    Assumes CSV has 12 columns corresponding to the 12 ECG leads.
    """
    print(f"Loading ECG data from {input_file}...")
    
    # Initialize variables
    lead_data = {lead: [] for lead in all_lead_names}
    data_line_count = 0
    
    try:
        with open(input_file, 'r', newline='') as csvfile:
            # Try to detect delimiter
            sample = csvfile.read(1024)
            csvfile.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            print(f"Detected delimiter: '{delimiter}'")
            
            reader = csv.reader(csvfile, delimiter=delimiter)
            
            for row_num, row in enumerate(reader, 1):
                # Skip empty rows
                if not row or all(cell.strip() == '' for cell in row):
                    continue
                
                # Check if we have exactly 12 values for the 12 leads
                if len(row) == 12:
                    data_line_count += 1
                    valid_row = True
                    row_values = []
                    
                    for i, value in enumerate(row):
                        try:
                            # Convert string to float
                            float_value = float(value.strip())
                            row_values.append(float_value)
                        except ValueError:
                            print(f"Warning: Invalid value '{value}' found at row {row_num}, column {i+1}")
                            # Use 0 as fallback
                            row_values.append(0.0)
                            valid_row = False
                    
                    # Add values to corresponding leads
                    for i, lead_name in enumerate(all_lead_names):
                        lead_data[lead_name].append(row_values[i])
                        
                elif len(row) > 0:  # Non-empty row with wrong number of columns
                    print(f"Warning: Expected 12 values but found {len(row)} at row {row_num}")
                    
        print(f"Processed {data_line_count} data rows")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found!")
        return None, None, 0
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None, 0
    
    # Verify all leads have the same length
    lead_lengths = {lead: len(data) for lead, data in lead_data.items()}
    unique_lengths = set(lead_lengths.values())
    
    if len(unique_lengths) > 1:
        print("Warning: Leads have different lengths!")
        for lead, length in lead_lengths.items():
            print(f"  {lead}: {length} samples")
        total_samples = min(lead_lengths.values())  # Use minimum length
        print(f"Using minimum length: {total_samples} samples")
        
        # Truncate all leads to minimum length
        for lead in lead_data:
            lead_data[lead] = lead_data[lead][:total_samples]
    else:
        total_samples = list(unique_lengths)[0] if unique_lengths else 0
        print(f"All leads have {total_samples} samples")
    
    return lead_data, default_sampling_rate, total_samples

def calculate_statistics(data):
    """Calculate basic statistics for a lead's data."""
    if not data:
        return {"min": 0, "max": 0, "mean": 0, "std": 0}
    
    np_data = np.array(data)
    return {
        "min": float(np.min(np_data)),
        "max": float(np.max(np_data)),
        "mean": float(np.mean(np_data)),
        "std": float(np.std(np_data))
    }

def main():
    # === Load and process ECG data ===
    leads, sampling_rate, num_samples_total = parse_csv_ecg_file(input_file)
    
    if leads is None:
        print("Failed to load ECG data. Exiting.")
        return
    
    if num_samples_total == 0:
        print("No valid data found. Exiting.")
        return
    
    # Calculate duration
    duration_seconds = num_samples_total / sampling_rate
    
    print(f"Total samples: {num_samples_total}")
    print(f"Duration: {duration_seconds:.2f} seconds")
    print(f"Sampling rate: {sampling_rate} Hz (default - update if known)")
    
    # === Calculate statistics for each lead ===
    lead_statistics = {}
    for lead_name, lead_data in leads.items():
        lead_statistics[lead_name] = calculate_statistics(lead_data)
    
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
            "source_file": input_file,
            "data_format": "CSV format",
            "note": "Sampling rate is default value - update if actual rate is known"
        },
        "statistics": lead_statistics,
        "leads": leads
    }
    
    # === Save to JSON file ===
    print(f"Saving to {output_file}...")
    try:
        with open(output_file, "w") as f:
            json.dump(complete_data, f, indent=2)
        
        print(f"Successfully saved complete ECG data to '{output_file}'")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        print(f"Leads included: {', '.join(leads.keys())}")
        
    except Exception as e:
        print(f"Error saving file: {e}")
        return
    
    # === Show data range summary ===
    print(f"\nData range summary:")
    
    measured_ranges = []
    derived_ranges = []
    
    for lead_name in all_lead_names:
        if lead_name in leads and len(leads[lead_name]) > 0:
            stats = lead_statistics[lead_name]
            range_str = f"{lead_name}: [{stats['min']:.3f}, {stats['max']:.3f}] mV (mean: {stats['mean']:.3f}, std: {stats['std']:.3f})"
            
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
        if any(abs(val) > 50 for val in data):  # Values > 50mV are suspicious for ECG
            max_abs = max(abs(val) for val in data)
            high_value_leads.append(f"{lead} (max: {max_abs:.3f} mV)")
    
    if high_value_leads:
        print(f"  Warning: Leads with suspiciously high values: {', '.join(high_value_leads)}")
    else:
        print("  ✓ No suspiciously high values detected")
    
    # Check for constant values (flat line)
    constant_leads = []
    for lead, data in leads.items():
        if len(set(data)) == 1:  # All values are the same
            constant_leads.append(f"{lead} (constant: {data[0]:.3f} mV)")
    
    if constant_leads:
        print(f"  Warning: Leads with constant values: {', '.join(constant_leads)}")
    else:
        print("  ✓ No leads with constant values")

if __name__ == "__main__":
    main()
