import xml.etree.ElementTree as ET
import json
import os
from pathlib import Path

# === Configuration ===
input_directory = "High RV leads"  # Change this to your input directory path
output_directory = "output_xml"  # Directory where all JSON files will be saved
file_group = "XML_High_RV"  # Prefix for output files

def create_output_filename(file_path, input_dir, file_group):
    """Create output filename with subdirectory path included"""
    # Get relative path from input directory
    rel_path = os.path.relpath(file_path, input_dir)
    
    # Get directory path and filename separately
    dir_path, filename = os.path.split(rel_path)
    
    # Remove only the .xml extension (case insensitive)
    if filename.lower().endswith('.xml'):
        base_name = filename[:-4]  # Remove last 4 characters (.xml)
    else:
        base_name = os.path.splitext(filename)[0]  # Fallback to splitext
    
    # Replace path separators with underscores and create output name
    if dir_path and dir_path != '.':
        # Replace both forward and backward slashes with underscores
        dir_str = dir_path.replace('/', '_').replace('\\', '_')
        output_name = f"{file_group}_{dir_str}_{base_name}.json"
    else:
        output_name = f"{file_group}_{base_name}.json"
    
    return output_name

def parse_xml_ecg_file(input_file):
    """
    Parse GE Sapphire ECG XML file and extract relevant data,
    including detailed metadata.
    """
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  ❌ Error parsing XML file: {e}")
        return None

    # Namespace for GE Sapphire XML
    namespace = {'ge': 'urn:ge:sapphire:sapphire_3'}

    # Initialize data structure
    ecg_data = {
        "metadata": {},
        "leads": {}
    }

    # === Extract General Metadata ===
    ecg_data["metadata"]["source_file"] = input_file
    ecg_data["metadata"]["data_format"] = "GE Sapphire XML format"

    # Extract top-level XML attributes like schema version
    if 'version' in root.attrib:
        ecg_data["metadata"]["schema_version"] = root.get('version')

    # Extract metadata from demographics section
    demographics = root.find('ge:demographics', namespace)
    if demographics is not None:
        session_id_element = demographics.find('ge:sessionID', namespace)
        if session_id_element is not None:
            ecg_data["metadata"]["session_id"] = session_id_element.get('V')

        patient_info = demographics.find('ge:patientInfo', namespace)
        if patient_info is not None:
            patient_id_element = patient_info.find("ge:identifier[@primary='true']", namespace)
            if patient_id_element is not None:
                ecg_data["metadata"]["patient_id"] = patient_id_element.get('V')

        test_info = demographics.find('ge:testInfo', namespace)
        if test_info is not None:
            acquisition_datetime_element = test_info.find('ge:acquisitionDateTime', namespace)
            if acquisition_datetime_element is not None:
                ecg_data["metadata"]["acquisition_datetime"] = acquisition_datetime_element.get('V')

    # === Extract Waveform Data and Sampling Rate ===
    ecg_block = root.find('.//ge:ecgWaveformMXG', namespace)
    if ecg_block is not None:
        sample_rate_element = ecg_block.find('ge:sampleRate', namespace)
        if sample_rate_element is not None:
            ecg_data["metadata"]["sampling_rate"] = int(sample_rate_element.get('V'))

        # Process each ecgWaveform element to get lead data
        for waveform_element in ecg_block.findall('ge:ecgWaveform', namespace):
            lead_name = waveform_element.get('lead')
            waveform_str = waveform_element.get('V')
            scale_str = waveform_element.get('S')
            unit = waveform_element.get('U') # Should be "uV" from XML

            if lead_name and waveform_str and scale_str and unit:
                try:
                    scale = float(scale_str)
                    # Convert raw values to millivolts (mV)
                    # Formula: (raw_value * S) / 1000 (since U is uV)
                    waveform_values = [float(val) * scale / 2500.0 for val in waveform_str.split()]
                    ecg_data["leads"][lead_name] = waveform_values
                except ValueError as e:
                    print(f"  ⚠️  Warning: Could not parse waveform data for lead {lead_name}: {e}")
    else:
        print("  ⚠️  Warning: ECG waveform data not found in the XML file.")

    # === Add Calculated Metadata (consistent with china_to_json.py) ===
    if ecg_data["leads"] and ecg_data["metadata"].get("sampling_rate"):
        # Assuming all leads have the same number of samples
        first_lead_name = next(iter(ecg_data["leads"]))
        total_samples = len(ecg_data["leads"][first_lead_name])
        ecg_data["metadata"]["total_samples"] = total_samples
        ecg_data["metadata"]["duration_seconds"] = total_samples / ecg_data["metadata"]["sampling_rate"]

    ecg_data["metadata"]["num_leads"] = len(ecg_data["leads"])
    ecg_data["metadata"]["lead_names"] = sorted(list(ecg_data["leads"].keys())) # Sorted for consistency
    ecg_data["metadata"]["units"] = "millivolts (mV)"

    # === Add Conversion Parameters (adapted for XML) ===
    # Based on BT="xs:short" (16-bit) and the S attribute in XML
    ecg_data["metadata"]["conversion_params"] = {
        "bit_depth": 16, # Inferred from BT="xs:short"
        "raw_data_type": "signed 16-bit integer",
        "raw_data_range": "(-32768 to 32767)",
        "intermediate_units_after_S_scaling": "microvolts (uV)",
        "final_output_units": "millivolts (mV)",
        "conversion_formula_description": "Each raw sample value is multiplied by its lead-specific 'S' (scale) attribute to get microvolts, then divided by 1000 to convert to millivolts."
    }

    return ecg_data

def process_xml_file(file_path, output_dir, input_dir, file_group):
    """Process a single XML file"""
    try:
        print(f"Processing: {file_path}")
        
        # Parse the XML file
        parsed_data = parse_xml_ecg_file(file_path)
        
        if not parsed_data:
            print(f"  ❌ Failed to parse {file_path}. Skipping...")
            return False
        
        # Check if we have leads data
        if not parsed_data.get("leads"):
            print(f"  ❌ No lead data found in {file_path}. Skipping...")
            return False
        
        # Create output filename
        output_filename = create_output_filename(file_path, input_dir, file_group)
        output_path = os.path.join(output_dir, output_filename)
        
        # Save JSON file
        with open(output_path, 'w') as f:
            json.dump(parsed_data, f, indent=4)
        
        # Show summary
        num_leads = len(parsed_data["leads"])
        duration = parsed_data["metadata"].get("duration_seconds", "unknown")
        sampling_rate = parsed_data["metadata"].get("sampling_rate", "unknown")
        
        print(f"  ✅ Saved to {output_filename}")
        print(f"     📊 {num_leads} leads, {duration}s duration, {sampling_rate}Hz sampling rate")
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {file_path}: {str(e)}")
        return False


def main():
    global file_group
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Ask user about file group prefix
    use_default_prefix = input(f"Use default file group prefix '{file_group}'? (y/n): ").lower().strip()
    
    if use_default_prefix != 'y':
        file_group = input("Enter file group prefix: ").strip() or file_group
    
    # Find all XML files recursively
    xml_files = []
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith('.xml'):
                xml_files.append(os.path.join(root, file))
    
    if not xml_files:
        print("❌ No XML files found in the specified directory.")
        return
    
    print(f"📁 Found {len(xml_files)} XML files to process...")
    print(f"📤 Output directory: {output_directory}")
    print(f"🏷️  File group prefix: {file_group}")
    print("-" * 50)
    
    # Process each file
    successful = 0
    failed = 0
    
    for file_path in xml_files:
        if process_xml_file(file_path, output_directory, input_directory, file_group):
            successful += 1
        else:
            failed += 1
    
    print("-" * 50)
    print(f"🎉 Processing complete!")
    print(f"✅ Successfully processed: {successful} files")
    print(f"❌ Failed: {failed} files")
    print(f"📁 All outputs saved in: {output_directory}")

if __name__ == "__main__":
    main()
