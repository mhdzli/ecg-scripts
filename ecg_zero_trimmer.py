import json
import os
from pathlib import Path

def trim_zeros_from_array(arr):
    """
    Remove leading and trailing zeros from an array.
    Returns the trimmed array.
    """
    if not arr:
        return arr
    
    # Find first non-zero index
    start_idx = 0
    for i, val in enumerate(arr):
        if val != 0.0 and val != 0:
            start_idx = i
            break
    else:
        # All values are zero
        return []
    
    # Find last non-zero index
    end_idx = len(arr) - 1
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] != 0.0 and arr[i] != 0:
            end_idx = i
            break
    
    return arr[start_idx:end_idx + 1]

def process_ecg_file(file_path):
    """
    Process a single ECG JSON file to remove leading/trailing zeros from leads.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if 'leads' key exists
        if 'leads' not in data:
            print(f"Warning: No 'leads' key found in {file_path}")
            return False
        
        # Process each lead
        leads_modified = False
        for lead_name, lead_data in data['leads'].items():
            if isinstance(lead_data, list):
                original_length = len(lead_data)
                trimmed_data = trim_zeros_from_array(lead_data)
                data['leads'][lead_name] = trimmed_data
                
                if len(trimmed_data) != original_length:
                    leads_modified = True
                    print(f"  {lead_name}: {original_length} -> {len(trimmed_data)} values")
        
        # Save the modified file
        if leads_modified:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        else:
            print(f"  No modifications needed")
            return False
            
    except json.JSONDecodeError as e:
        print(f"Error reading JSON from {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def process_directory(directory_path, backup=True):
    """
    Process all JSON files in a directory.
    
    Args:
        directory_path (str): Path to directory containing JSON files
        backup (bool): Whether to create backup files before modification
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Directory {directory_path} does not exist!")
        return
    
    # Find all JSON files
    json_files = list(directory.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    processed_count = 0
    modified_count = 0
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")
        
        # Create backup if requested
        if backup:
            backup_path = json_file.with_suffix('.json.backup')
            if not backup_path.exists():
                json_file.rename(backup_path)
                json_file.write_text(backup_path.read_text())
        
        # Process the file
        was_modified = process_ecg_file(json_file)
        processed_count += 1
        
        if was_modified:
            modified_count += 1
    
    print(f"\nSummary:")
    print(f"  Files processed: {processed_count}")
    print(f"  Files modified: {modified_count}")
    print(f"  Files unchanged: {processed_count - modified_count}")

# Example usage
if __name__ == "__main__":
    # Specify your directory path here
    directory_path = "converted_json"  # Change this to your actual directory
    
    # Process all JSON files in the directory
    # Set backup=False if you don't want backup files created
    process_directory(directory_path, backup=True)
    
    # Alternative: Process a single file
    # process_ecg_file("path/to/single/file.json")
