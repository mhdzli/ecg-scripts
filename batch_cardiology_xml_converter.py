import xml.etree.ElementTree as ET
import json
import os
from pathlib import Path

# === Configuration ===
input_directory = "RASE digital"  # Change this to your input directory path
output_directory = "new_output"  # Directory where all JSON files will be saved
file_group = "XML_RASE"  # Prefix for output files

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

def safe_get_text(element):
    """Safely get text from XML element, return empty string if None"""
    if element is not None:
        return element.text.strip() if element.text else ""
    return ""

def parse_cardiology_xml_file(input_file):
    """
    Parse CardiologyXML file and extract relevant data,
    including detailed metadata.
    """
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  ❌ Error parsing XML file: {e}")
        return None

    # Initialize data structure
    ecg_data = {
        "metadata": {},
        "leads": {}
    }

    # === Extract General Metadata ===
    ecg_data["metadata"]["source_file"] = str(input_file)
    ecg_data["metadata"]["data_format"] = "CardiologyXML format"

    # Extract observation type
    obs_type = root.find('ObservationType')
    if obs_type is not None:
        ecg_data["metadata"]["observation_type"] = safe_get_text(obs_type)

    # Extract observation date and time
    obs_datetime = root.find('ObservationDateTime')
    if obs_datetime is not None:
        try:
            year = safe_get_text(obs_datetime.find('Year'))
            month = safe_get_text(obs_datetime.find('Month'))
            day = safe_get_text(obs_datetime.find('Day'))
            hour = safe_get_text(obs_datetime.find('Hour'))
            minute = safe_get_text(obs_datetime.find('Minute'))
            second = safe_get_text(obs_datetime.find('Second'))
            
            if all([year, month, day, hour, minute, second]):
                ecg_data["metadata"]["acquisition_datetime"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}T{hour.zfill(2)}:{minute.zfill(2)}:{second.zfill(2)}"
        except:
            pass

    # Extract UID information
    uid_elem = root.find('UID/DICOMStudyUID')
    if uid_elem is not None:
        ecg_data["metadata"]["dicom_study_uid"] = safe_get_text(uid_elem)

    # Extract device information
    device_info = root.find('ClinicalInfo/DeviceInfo')
    if device_info is not None:
        desc = device_info.find('Desc')
        if desc is not None:
            ecg_data["metadata"]["device_description"] = safe_get_text(desc)
        
        software_ver = device_info.find('SoftwareVer')
        if software_ver is not None:
            ecg_data["metadata"]["software_version"] = safe_get_text(software_ver)
        
        analysis_ver = device_info.find('AnalysisVer')
        if analysis_ver is not None:
            ecg_data["metadata"]["analysis_version"] = safe_get_text(analysis_ver)

    # Extract patient information
    patient_info = root.find('PatientInfo')
    if patient_info is not None:
        pid = patient_info.find('PID')
        if pid is not None:
            ecg_data["metadata"]["patient_id"] = safe_get_text(pid)
        
        # Extract patient name
        name = patient_info.find('Name')
        if name is not None:
            given_name = safe_get_text(name.find('GivenName'))
            family_name = safe_get_text(name.find('FamilyName'))
            if given_name or family_name:
                ecg_data["metadata"]["patient_name"] = f"{given_name} {family_name}".strip()
        
        # Extract other patient demographics
        age = patient_info.find('Age')
        if age is not None:
            ecg_data["metadata"]["patient_age"] = safe_get_text(age)
            ecg_data["metadata"]["patient_age_units"] = age.get('units', '')
        
        gender = patient_info.find('Gender')
        if gender is not None:
            ecg_data["metadata"]["patient_gender"] = safe_get_text(gender)
        
        height = patient_info.find('Height')
        if height is not None:
            ecg_data["metadata"]["patient_height"] = safe_get_text(height)
            ecg_data["metadata"]["patient_height_units"] = height.get('units', '')
        
        weight = patient_info.find('Weight')
        if weight is not None:
            ecg_data["metadata"]["patient_weight"] = safe_get_text(weight)
            ecg_data["metadata"]["patient_weight_units"] = weight.get('units', '')

    # Extract filter settings
    filter_setting = root.find('FilterSetting')
    if filter_setting is not None:
        ecg_data["metadata"]["filter_settings"] = {}
        for filter_elem in filter_setting:
            if filter_elem.tag in ['LowPass', 'HighPass']:
                ecg_data["metadata"]["filter_settings"][filter_elem.tag.lower()] = {
                    "value": safe_get_text(filter_elem),
                    "units": filter_elem.get('units', '')
                }
            else:
                ecg_data["metadata"]["filter_settings"][filter_elem.tag.lower()] = safe_get_text(filter_elem)

    # Extract interpretation/diagnosis
    interpretation = root.find('Interpretation')
    if interpretation is not None:
        diagnosis_texts = []
        for diag in interpretation.findall('Diagnosis/DiagnosisText'):
            diag_text = safe_get_text(diag)
            if diag_text:
                diagnosis_texts.append(diag_text)
        if diagnosis_texts:
            ecg_data["metadata"]["diagnosis"] = diagnosis_texts

    # === Extract Waveform Data ===
    strip_data = root.find('StripData')
    if strip_data is not None:
        # Extract sampling parameters
        sample_rate_elem = strip_data.find('SampleRate')
        if sample_rate_elem is not None:
            ecg_data["metadata"]["sampling_rate"] = int(safe_get_text(sample_rate_elem))
        
        num_leads_elem = strip_data.find('NumberOfLeads')
        if num_leads_elem is not None:
            ecg_data["metadata"]["number_of_leads"] = int(safe_get_text(num_leads_elem))
        
        total_samples_elem = strip_data.find('ChannelSampleCountTotal')
        if total_samples_elem is not None:
            ecg_data["metadata"]["total_samples"] = int(safe_get_text(total_samples_elem))
        
        resolution_elem = strip_data.find('Resolution')
        if resolution_elem is not None:
            ecg_data["metadata"]["resolution"] = float(safe_get_text(resolution_elem))
            ecg_data["metadata"]["resolution_units"] = resolution_elem.get('units', '')

        # Process waveform data
        for waveform_elem in strip_data.findall('WaveformData'):
            lead_name = waveform_elem.get('lead')
            waveform_str = safe_get_text(waveform_elem)
            
            if lead_name and waveform_str:
                try:
                    # Parse comma-separated values
                    raw_values = [int(val.strip()) for val in waveform_str.split(',') if val.strip()]
                    
                    # Convert to millivolts using resolution
                    # Resolution is in uV per LSB, so multiply by resolution then divide by 1000 to get mV
                    resolution = ecg_data["metadata"].get("resolution", 5.0)  # Default to 5 uV/LSB
                    waveform_values = [float(val) * resolution / 1000.0 for val in raw_values]
                    
                    ecg_data["leads"][lead_name] = waveform_values
                except (ValueError, TypeError) as e:
                    print(f"  ⚠️  Warning: Could not parse waveform data for lead {lead_name}: {e}")

    # === Add Calculated Metadata ===
    if ecg_data["leads"]:
        # Get actual sample count from data
        first_lead_name = next(iter(ecg_data["leads"]))
        actual_samples = len(ecg_data["leads"][first_lead_name])
        ecg_data["metadata"]["actual_samples_per_lead"] = actual_samples
        
        # Calculate duration
        if ecg_data["metadata"].get("sampling_rate"):
            ecg_data["metadata"]["duration_seconds"] = actual_samples / ecg_data["metadata"]["sampling_rate"]

    ecg_data["metadata"]["num_leads"] = len(ecg_data["leads"])
    ecg_data["metadata"]["lead_names"] = sorted(list(ecg_data["leads"].keys()))
    ecg_data["metadata"]["units"] = "millivolts (mV)"

    # === Add Conversion Parameters ===
    ecg_data["metadata"]["conversion_params"] = {
        "raw_data_type": "signed integer",
        "raw_data_format": "comma-separated values",
        "resolution_description": "microvolts per least significant bit (uV/LSB)",
        "intermediate_units_after_resolution_scaling": "microvolts (uV)",
        "final_output_units": "millivolts (mV)",
        "conversion_formula_description": "Each raw sample value is multiplied by the resolution (uV/LSB) to get microvolts, then divided by 1000 to convert to millivolts."
    }

    return ecg_data

def process_xml_file(file_path, output_dir, input_dir, file_group):
    """Process a single XML file"""
    try:
        print(f"Processing: {file_path}")
        
        # Parse the XML file
        parsed_data = parse_cardiology_xml_file(file_path)
        
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
