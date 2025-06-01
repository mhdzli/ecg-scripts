import xml.etree.ElementTree as ET
import json
import os

def parse_xml_ecg_file(input_file):
    """
    Parse GE Sapphire ECG XML file and extract relevant data,
    including detailed metadata.
    """
    print(f"Loading ECG data from {input_file}...")

    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None

    # Namespace for GE Sapphire XML
    namespace = {'ge': 'urn:ge:sapphire:sapphire_3'}

    # Initialize data structure
    ecg_data = {
        "metadata": {},
        "leads": {}
    }

    # Extract top-level XML attributes like schema version
    if 'version' in root.attrib:
        ecg_data["metadata"]["schema_version"] = root.get('version')

    # Extract metadata (patient info, acquisition datetime, session ID)
    demographics = root.find('ge:demographics', namespace)
    if demographics is not None:
        # Session ID
        session_id_element = demographics.find('ge:sessionID', namespace)
        if session_id_element is not None:
            ecg_data["metadata"]["session_id"] = session_id_element.get('V')

        # Patient Information
        patient_info = demographics.find('ge:patientInfo', namespace)
        if patient_info is not None:
            # Primary patient identifier
            patient_id_element = patient_info.find("ge:identifier[@primary='true']", namespace)
            if patient_id_element is not None:
                ecg_data["metadata"]["patient_id"] = patient_id_element.get('V')
            
            # Additional patient details can be added here if needed, e.g., name, race, etc.
            # For example, patient_name_element = patient_info.find('ge:name/ge:given', namespace)

        # Test Information (e.g., acquisition date/time)
        test_info = demographics.find('ge:testInfo', namespace)
        if test_info is not None:
            acquisition_datetime_element = test_info.find('ge:acquisitionDateTime', namespace)
            if acquisition_datetime_element is not None:
                ecg_data["metadata"]["acquisition_datetime"] = acquisition_datetime_element.get('V')
            
            # Other test-related metadata could be extracted here

    # Extract waveform data and sampling rate
    ecg_block = root.find('.//ge:ecgWaveformMXG', namespace)
    if ecg_block is not None:
        sample_rate_element = ecg_block.find('ge:sampleRate', namespace)
        if sample_rate_element is not None:
            # Place sampling_rate directly within metadata
            ecg_data["metadata"]["sampling_rate"] = int(sample_rate_element.get('V'))

        # Iterate through each ecgWaveform element to get lead data
        for waveform_element in ecg_block.findall('ge:ecgWaveform', namespace):
            lead_name = waveform_element.get('lead')
            waveform_str = waveform_element.get('V')
            scale_str = waveform_element.get('S')
            unit = waveform_element.get('U')

            if lead_name and waveform_str and scale_str and unit:
                try:
                    scale = float(scale_str)
                    # Convert the space-separated string of values to a list of floats.
                    # The XML provides data in microvolts (uV), converting to millivolts (mV)
                    # by dividing by 1000 for consistency.
                    waveform_values = [float(val) * scale / 1000.0 for val in waveform_str.split()]
                    ecg_data["leads"][lead_name] = waveform_values
                except ValueError as e:
                    print(f"Warning: Could not parse waveform data for lead {lead_name}: {e}")
    else:
        print("Warning: ECG waveform data not found in the XML file.")

    return ecg_data

# === Configuration ===
file_group = "ge_xml"
input_xml_file = "BRS37__20240725_104753.Xml"  # Make sure this path is correct
output_json_file = f"{file_group}_{os.path.splitext(input_xml_file)[0]}.json"

def main():
    parsed_data = parse_xml_ecg_file(input_xml_file)

    if parsed_data:
        try:
            with open(output_json_file, 'w') as f:
                json.dump(parsed_data, f, indent=4) # Use indent for pretty printing
            print(f"\nSuccessfully converted {input_xml_file} to {output_json_file}")
        except IOError as e:
            print(f"Error writing JSON file: {e}")
    else:
        print(f"Failed to parse {input_xml_file}. No JSON output generated.")

if __name__ == "__main__":
    main()