import xml.etree.ElementTree as ET
import json
import os
from pathlib import Path

# === Configuration ===
input_directory = "RBH"  # Change this to your input directory path
output_directory = "output"  # Directory where all JSON files will be saved
file_group = "XML_FDA_RBH"  # Prefix for output files

# Fixed parameters for the new XML format
SAMPLING_RATE = 500  # Hz - fixed for this format
ADC_RESOLUTION_DIVISOR = 2500  # scale/2500 as per your specification

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

def parse_hl7_xml_ecg_file(input_file):
    """
    Parse HL7 AnnotatedECG XML file and extract relevant data,
    including detailed metadata.
    """
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  ❌ Error parsing XML file: {e}")
        return None

    # Namespace for HL7 AnnotatedECG XML
    namespace = {'hl7': 'urn:hl7-org:v3'}

    # Initialize data structure
    ecg_data = {
        "metadata": {},
        "leads": {}
    }

    # === Extract General Metadata ===
    ecg_data["metadata"]["source_file"] = input_file
    ecg_data["metadata"]["data_format"] = "HL7 AnnotatedECG XML format"
    ecg_data["metadata"]["sampling_rate"] = SAMPLING_RATE

    # Extract ECG ID
    ecg_id_element = root.find('hl7:id', namespace)
    if ecg_id_element is not None:
        ecg_data["metadata"]["ecg_id"] = ecg_id_element.get('root')

    # Extract ECG code
    code_element = root.find('hl7:code', namespace)
    if code_element is not None:
        ecg_data["metadata"]["ecg_code"] = code_element.get('code')

    # Extract effective time
    effective_time_element = root.find('hl7:effectiveTime/hl7:center', namespace)
    if effective_time_element is not None:
        ecg_data["metadata"]["effective_time"] = effective_time_element.get('value')

    # Extract patient demographics
    subject_element = root.find('.//hl7:trialSubject', namespace)
    if subject_element is not None:
        subject_id_element = subject_element.find('hl7:id', namespace)
        if subject_id_element is not None:
            ecg_data["metadata"]["subject_id"] = subject_id_element.get('root')
            ecg_data["metadata"]["subject_extension"] = subject_id_element.get('extension')

        # Patient demographics
        demographics_element = subject_element.find('hl7:subjectDemographicPerson', namespace)
        if demographics_element is not None:
            gender_element = demographics_element.find('hl7:administrativeGenderCode', namespace)
            if gender_element is not None:
                ecg_data["metadata"]["gender"] = gender_element.get('code')

            birth_time_element = demographics_element.find('hl7:birthTime', namespace)
            if birth_time_element is not None:
                ecg_data["metadata"]["birth_time"] = birth_time_element.get('value')

    # Extract clinical trial information
    trial_element = root.find('.//hl7:clinicalTrial', namespace)
    if trial_element is not None:
        trial_id_element = trial_element.find('hl7:id', namespace)
        if trial_id_element is not None:
            ecg_data["metadata"]["trial_id"] = trial_id_element.get('root')

        # Trial site location
        location_element = trial_element.find('.//hl7:location/hl7:name', namespace)
        if location_element is not None:
            ecg_data["metadata"]["location"] = location_element.text

    # Extract device information
    device_element = root.find('.//hl7:manufacturedSeriesDevice', namespace)
    if device_element is not None:
        device_id_element = device_element.find('hl7:id', namespace)
        if device_id_element is not None:
            ecg_data["metadata"]["device_id"] = device_id_element.get('extension')

        device_code_element = device_element.find('hl7:code', namespace)
        if device_code_element is not None:
            ecg_data["metadata"]["device_code"] = device_code_element.get('code')

        device_name_element = device_element.find('hl7:manufacturerModelName', namespace)
        if device_name_element is not None:
            ecg_data["metadata"]["device_name"] = device_name_element.text

    # Extract filter settings
    control_variables = root.findall('.//hl7:controlVariable', namespace)
    filters = {}
    for cv in control_variables:
        code_element = cv.find('hl7:code', namespace)
        if code_element is not None:
            code = code_element.get('code')
            display_name = code_element.get('displayName')
            
            value_element = cv.find('hl7:value', namespace)
            if value_element is not None:
                value = value_element.get('value')
                unit = value_element.get('unit')
                filters[display_name or code] = {"value": value, "unit": unit}
            
            # Check for nested filter components (like cutoff frequency)
            component_cv = cv.find('.//hl7:controlVariable', namespace)
            if component_cv is not None:
                comp_code_element = component_cv.find('hl7:code', namespace)
                comp_value_element = component_cv.find('hl7:value', namespace)
                if comp_code_element is not None and comp_value_element is not None:
                    comp_display_name = comp_code_element.get('displayName')
                    comp_value = comp_value_element.get('value')
                    comp_unit = comp_value_element.get('unit')
                    filters[f"{display_name or code} - {comp_display_name}"] = {
                        "value": comp_value, 
                        "unit": comp_unit
                    }

    if filters:
        ecg_data["metadata"]["filters"] = filters

    # === Extract Waveform Data ===
    sequence_set = root.find('.//hl7:sequenceSet', namespace)
    if sequence_set is not None:
        sequences = sequence_set.findall('hl7:component/hl7:sequence', namespace)
        
        for sequence in sequences:
            code_element = sequence.find('hl7:code', namespace)
            if code_element is not None:
                lead_code = code_element.get('code')
                
                # Skip the TIME_ABSOLUTE sequence
                if lead_code == "TIME_ABSOLUTE":
                    continue
                
                # Extract lead name from code (e.g., "MDC_ECG_LEAD_I" -> "I")
                if lead_code.startswith("MDC_ECG_LEAD_"):
                    lead_name = lead_code.replace("MDC_ECG_LEAD_", "")
                else:
                    lead_name = lead_code
                
                # Extract waveform data
                value_element = sequence.find('hl7:value', namespace)
                if value_element is not None:
                    origin_element = value_element.find('hl7:origin', namespace)
                    scale_element = value_element.find('hl7:scale', namespace)
                    if scale_element is not None:
                        scale = float(scale_element.get('value', 1))
                        unit = scale_element.get('unit', 'uV')
                    if origin_element is not None:
                        origin = float(origin_element.get('value', 0))/1000
                    
                    digits_element = value_element.find('hl7:digits', namespace)
                    if digits_element is not None and digits_element.text:
                        try:
                            # Parse the digits string to get raw values
                            raw_values = [int(x) for x in digits_element.text.split()]
                            
                            # Apply conversion: (raw_value * scale) / ADC_RESOLUTION_DIVISOR
                            # This converts to millivolts as per your specification
                            converted_values = [
                                (origin + (raw_value * scale) / ADC_RESOLUTION_DIVISOR) 
                                for raw_value in raw_values
                            ]
                            
                            if lead_name in {"AVF", "AVL", "AVR"}:
                                lead_name = "a" + lead_name[1:]
                            ecg_data["leads"][lead_name] = converted_values
                            
                        except ValueError as e:
                            print(f"  ⚠️  Warning: Could not parse digits for lead {lead_name}: {e}")
    
    # === Extract Annotations (Heart Rate, Intervals, etc.) ===
    annotations = {}
    annotation_set = root.find('.//hl7:annotationSet', namespace)
    if annotation_set is not None:
        annotation_components = annotation_set.findall('hl7:component/hl7:annotation', namespace)
        
        for annotation in annotation_components:
            code_element = annotation.find('hl7:code', namespace)
            if code_element is not None:
                code = code_element.get('code')
                display_name = code_element.get('displayName')
                
                # Handle simple value annotations
                value_element = annotation.find('hl7:value', namespace)
                if value_element is not None:
                    value = value_element.get('value')
                    unit = value_element.get('unit')
                    annotations[display_name or code] = {"value": value, "unit": unit}
                
                # Handle nested annotations (like interpretations)
                nested_components = annotation.findall('hl7:component/hl7:annotation', namespace)
                for nested in nested_components:
                    nested_code_element = nested.find('hl7:code', namespace)
                    nested_value_element = nested.find('hl7:value', namespace)
                    
                    if nested_code_element is not None and nested_value_element is not None:
                        nested_code = nested_code_element.get('code')
                        nested_display_name = nested_code_element.get('displayName')
                        nested_value = nested_value_element.text
                        
                        if nested_value:
                            annotations[nested_display_name or nested_code] = nested_value

    if annotations:
        ecg_data["metadata"]["annotations"] = annotations

    # === Add Calculated Metadata ===
    if ecg_data["leads"]:
        # Assuming all leads have the same number of samples
        first_lead_name = next(iter(ecg_data["leads"]))
        total_samples = len(ecg_data["leads"][first_lead_name])
        ecg_data["metadata"]["total_samples"] = total_samples
        ecg_data["metadata"]["duration_seconds"] = total_samples / SAMPLING_RATE

    ecg_data["metadata"]["num_leads"] = len(ecg_data["leads"])
    ecg_data["metadata"]["lead_names"] = sorted(list(ecg_data["leads"].keys()))
    ecg_data["metadata"]["units"] = "millivolts (mV)"

    # === Add Conversion Parameters ===
    ecg_data["metadata"]["conversion_params"] = {
        "adc_resolution_divisor": ADC_RESOLUTION_DIVISOR,
        "sampling_rate": SAMPLING_RATE,
        "conversion_formula": "converted_value = (raw_value * scale) / 2500",
        "raw_data_units": "ADC counts",
        "intermediate_units": "microvolts (uV) after scale multiplication",
        "final_output_units": "millivolts (mV)"
    }

    return ecg_data

def process_xml_file(file_path, output_dir, input_dir, file_group):
    """Process a single XML file"""
    try:
        print(f"Processing: {file_path}")
        
        # Parse the XML file
        parsed_data = parse_hl7_xml_ecg_file(file_path)
        
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
        
        # Show annotations if available
        annotations = parsed_data["metadata"].get("annotations", {})
        if annotations:
            heart_rate = annotations.get("MDC_ECG_HEART_RATE", {}).get("value", "N/A")
            print(f"     💓 Heart rate: {heart_rate} bpm")
        
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
    print(f"⚙️  Sampling rate: {SAMPLING_RATE} Hz")
    print(f"🔧 ADC resolution: scale/{ADC_RESOLUTION_DIVISOR}")
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
