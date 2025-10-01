import os
import json
import csv
import numpy as np
from pathlib import Path

# === Configuration ===
file_group = "CSV_RASE_pdf_digitised"
input_directory = "RASE pdf (digitised)"  # Current directory - change this to your input directory
output_directory = "converted_json"  # Directory to store all output files
csv_extensions = ['.csv']  # File extensions to process

# Default sampling rate (update if known)
default_sampling_rate = 1000  # Hz - adjust if you know the actual sampling rate

# All 12 standard ECG leads (in the order they appear in your CSV)
all_lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
measured_leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']  # Typically measured
derived_leads = ['III', 'aVR', 'aVL', 'aVF']  # Typically derived
adc_resulution = 1/2.5

def create_output_filename(file_path, input_dir, output_dir):
    """
    Create output filename with subdirectory path information.
    Example: subdir1/subdir2/filename.csv -> CSV_ECG_subdir1_subdir2_filename.json
    """
    # Get relative path from input directory
    rel_path = os.path.relpath(file_path, input_dir)
    
    # Split path into parts
    path_parts = Path(rel_path).parts
    
    # Remove file extension from the last part (filename)
    filename_without_ext = Path(path_parts[-1]).stem
    
    # Create path string for subdirectories (exclude filename)
    if len(path_parts) > 1:
        subdir_parts = path_parts[:-1]
        subdir_string = "_".join(subdir_parts)
        output_name = f"{file_group}_{subdir_string}_{filename_without_ext}.json"
    else:
        output_name = f"{file_group}_{filename_without_ext}.json"
    
    # Replace spaces and special characters with underscores
    output_name = output_name.replace(' ', '_').replace('-', '_')
    
    # Create full output path
    return os.path.join(output_dir, output_name)

def find_csv_files(directory):
    """Find all CSV files in directory and subdirectories."""
    csv_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in csv_extensions):
                csv_files.append(os.path.join(root, file))
    
    return csv_files

def parse_csv_ecg_file(input_file):
    """
    Parse CSV ECG file and extract lead data.
    Assumes CSV has 12 columns corresponding to the 12 ECG leads.
    """
    print(f"  Loading ECG data from {input_file}...")
    
    # Initialize variables
    lead_data = {lead: [] for lead in all_lead_names}
    data_line_count = 0
    
    try:
        with open(input_file, 'r', newline='', encoding='utf-8', errors='ignore') as csvfile:
            # Read first few lines to better detect format
            first_lines = []
            for i in range(5):
                try:
                    line = csvfile.readline()
                    if line:
                        first_lines.append(line.strip())
                    else:
                        break
                except:
                    break
            
            # Reset file pointer
            csvfile.seek(0)
            
            # Try different delimiters and parsing approaches
            delimiters_to_try = [',', ';', '\t', '|']
            best_delimiter = ','
            max_columns = 0
            
            # Test each delimiter on the first few lines
            for delimiter in delimiters_to_try:
                column_counts = []
                for line in first_lines:
                    if line.strip():
                        parts = line.split(delimiter)
                        column_counts.append(len(parts))
                
                if column_counts:
                    avg_columns = sum(column_counts) / len(column_counts)
                    if avg_columns > max_columns and avg_columns >= 12:
                        max_columns = avg_columns
                        best_delimiter = delimiter
            
            print(f"    Using delimiter: '{best_delimiter}', detected ~{max_columns:.0f} columns")
            
            # Reset file pointer again
            csvfile.seek(0)
            
            # Method 1: Try using csv.reader
            try:
                reader = csv.reader(csvfile, delimiter=best_delimiter)
                
                for row_num, row in enumerate(reader, 1):
                    # Skip empty rows
                    if not row or all(cell.strip() == '' for cell in row):
                        continue
                    
                    # Remove any trailing empty cells
                    while row and row[-1].strip() == '':
                        row.pop()
                    
                    # Check if we have exactly 12 values for the 12 leads
                    if len(row) == 12:
                        data_line_count += 1
                        row_values = []
                        
                        for i, value in enumerate(row):
                            try:
                                # Convert string to float
                                float_value = float(value.strip())
                                row_values.append(float_value)
                            except ValueError:
                                # Use 0 as fallback for invalid values
                                row_values.append(0.0)
                        
                        # Add values to corresponding leads
                        for i, lead_name in enumerate(all_lead_names):
                            lead_data[lead_name].append(row_values[i] * adc_resulution)
                            
                    elif len(row) > 12:
                        # Take only first 12 values if there are more
                        data_line_count += 1
                        row_values = []
                        
                        for i in range(12):
                            try:
                                float_value = float(row[i].strip())
                                row_values.append(float_value)
                            except (ValueError, IndexError):
                                row_values.append(0.0)
                        
                        # Add values to corresponding leads
                        for i, lead_name in enumerate(all_lead_names):
                            lead_data[lead_name].append(row_values[i] * adc_resulution)
                            
                    elif len(row) > 0:  # Non-empty row with wrong number of columns
                        if row_num <= 5:  # Only show warnings for first few rows
                            print(f"    Warning: Expected 12 values but found {len(row)} at row {row_num}: {row[:3]}...")
                        
                        # If we have fewer than 12 but more than 0, try to pad or use what we have
                        if len(row) >= 6:  # At least half the expected columns
                            data_line_count += 1
                            row_values = []
                            
                            for i in range(12):
                                try:
                                    if i < len(row):
                                        float_value = float(row[i].strip())
                                        row_values.append(float_value)
                                    else:
                                        row_values.append(0.0)  # Pad with zeros
                                except ValueError:
                                    row_values.append(0.0)
                            
                            # Add values to corresponding leads
                            for i, lead_name in enumerate(all_lead_names):
                                lead_data[lead_name].append(row_values[i] * adc_resulution)
                
            except Exception as csv_error:
                print(f"    CSV reader failed: {csv_error}")
                print(f"    Trying manual line parsing...")
                
                # Method 2: Manual line-by-line parsing as fallback
                csvfile.seek(0)
                for row_num, line in enumerate(csvfile, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Split by best delimiter
                    parts = line.split(best_delimiter)
                    
                    # Remove empty parts at the end
                    while parts and parts[-1].strip() == '':
                        parts.pop()
                    
                    if len(parts) >= 12:
                        data_line_count += 1
                        row_values = []
                        
                        for i in range(12):
                            try:
                                float_value = float(parts[i].strip())
                                row_values.append(float_value)
                            except (ValueError, IndexError):
                                row_values.append(0.0)
                        
                        # Add values to corresponding leads
                        for i, lead_name in enumerate(all_lead_names):
                            lead_data[lead_name].append(row_values[i] * adc_resulution)
                    
                    elif len(parts) > 0 and row_num <= 5:
                        print(f"    Warning: Expected 12 values but found {len(parts)} at row {row_num}")
                        
        print(f"    Processed {data_line_count} data rows")
        
        # Debug: Print first few values of each lead
        if data_line_count > 0:
            print(f"    Sample data (first 3 samples):")
            for lead_name in all_lead_names[:4]:  # Show first 4 leads
                if lead_data[lead_name]:
                    sample_values = lead_data[lead_name][:3]
                    print(f"      {lead_name}: {sample_values}")
        
    except FileNotFoundError:
        print(f"    Error: File '{input_file}' not found!")
        return None, None, 0
    except Exception as e:
        print(f"    Error reading file: {e}")
        return None, None, 0
    
    # Verify all leads have the same length
    lead_lengths = {lead: len(data) for lead, data in lead_data.items()}
    unique_lengths = set(lead_lengths.values())
    
    if len(unique_lengths) > 1:
        total_samples = min(lead_lengths.values())  # Use minimum length
        print(f"    Warning: Leads have different lengths. Using minimum: {total_samples}")
        # Truncate all leads to minimum length
        for lead in lead_data:
            lead_data[lead] = lead_data[lead][:total_samples]
    else:
        total_samples = list(unique_lengths)[0] if unique_lengths else 0
    
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

def process_single_file(input_file, output_file):
    """Process a single CSV file and convert to JSON."""
    # === Load and process ECG data ===
    leads, sampling_rate, num_samples_total = parse_csv_ecg_file(input_file)
    
    if leads is None or num_samples_total == 0:
        print(f"    Failed to load valid ECG data from {input_file}")
        return False
    
    # Calculate duration
    duration_seconds = num_samples_total / sampling_rate
    
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
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(complete_data, f, indent=2)
        
        file_size_mb = os.path.getsize(output_file) / (1024*1024)
        print(f"    ✓ Saved to {output_file} ({file_size_mb:.2f} MB)")
        print(f"    Duration: {duration_seconds:.2f}s, Samples: {num_samples_total}")
        
        return True
        
    except Exception as e:
        print(f"    Error saving file: {e}")
        return False

def perform_data_quality_check(leads, filename):
    """Perform basic data quality checks."""
    issues = []
    
    # Check for leads with all zeros
    zero_leads = [lead for lead, data in leads.items() if all(val == 0.0 for val in data)]
    if zero_leads:
        issues.append(f"All-zero leads: {', '.join(zero_leads)}")
    
    # Check for extremely high values (potential parsing errors)
    high_value_leads = []
    for lead, data in leads.items():
        if any(abs(val) > 50 for val in data):  # Values > 50mV are suspicious for ECG
            max_abs = max(abs(val) for val in data)
            high_value_leads.append(f"{lead}({max_abs:.1f}mV)")
    
    if high_value_leads:
        issues.append(f"High values: {', '.join(high_value_leads)}")
    
    # Check for constant values (flat line)
    constant_leads = [lead for lead, data in leads.items() if len(set(data)) == 1]
    if constant_leads:
        issues.append(f"Constant: {', '.join(constant_leads)}")
    
    if issues:
        print(f"    ⚠ Quality issues: {'; '.join(issues)}")

def main():
    print(f"=== Batch CSV to JSON ECG Converter (Fixed Version) ===")
    print(f"Input directory: {os.path.abspath(input_directory)}")
    print(f"Output directory: {os.path.abspath(output_directory)}")
    print(f"File extensions: {', '.join(csv_extensions)}")
    print()
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    # Find all CSV files
    csv_files = find_csv_files(input_directory)
    
    if not csv_files:
        print("No CSV files found in the specified directory and subdirectories.")
        return
    
    print(f"Found {len(csv_files)} CSV files to process:")
    for i, file_path in enumerate(csv_files, 1):
        rel_path = os.path.relpath(file_path, input_directory)
        print(f"  {i:3d}. {rel_path}")
    print()
    
    # Process each file
    successful_conversions = 0
    failed_conversions = 0
    
    for i, input_file in enumerate(csv_files, 1):
        rel_path = os.path.relpath(input_file, input_directory)
        output_file = create_output_filename(input_file, input_directory, output_directory)
        
        print(f"[{i:3d}/{len(csv_files)}] Processing: {rel_path}")
        print(f"    Output: {os.path.basename(output_file)}")
        
        success = process_single_file(input_file, output_file)
        
        if success:
            successful_conversions += 1
            
            # Load the data back for quality check
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    perform_data_quality_check(data['leads'], input_file)
            except:
                pass  # Skip quality check if there's an issue
        else:
            failed_conversions += 1
        
        print()
    
    # Summary
    print("=== Conversion Summary ===")
    print(f"Total files processed: {len(csv_files)}")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    print(f"Output directory: {os.path.abspath(output_directory)}")
    
    if successful_conversions > 0:
        print(f"\n✓ All converted JSON files are stored in: {os.path.abspath(output_directory)}")

if __name__ == "__main__":
    main()
