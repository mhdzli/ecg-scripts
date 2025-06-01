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

    # === Extract General Metadata ===
    ecg_data["metadata"]["source_file"] = os.path.basename(input_file)
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
                    waveform_values = [float(val) * scale / 1000.0 for val in waveform_str.split()]
                    ecg_data["leads"][lead_name] = waveform_values
                except ValueError as e:
                    print(f"Warning: Could not parse waveform data for lead {lead_name}: {e}")
    else:
        print("Warning: ECG waveform data not found in the XML file.")

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

# === Configuration ===
file_group = "ge"
input_xml_file = "BRS37__20240725_104753.Xml"  # Set this to the path of your XML file
output_json_file = f"{file_group}_{os.path.splitext(input_xml_file)[0]}.json"

def main():
    parsed_data = parse_xml_ecg_file(input_xml_file)

    if parsed_data:
        try:
            with open(output_json_file, 'w') as f:
                json.dump(parsed_data, f, indent=4) # Use indent for pretty printing
            print(f"\nSuccessfully converted {input_xml_file} to {output_json_file}")
            print(f"Output saved to: {os.path.abspath(output_json_file)}")
        except IOError as e:
            print(f"Error writing JSON file: {e}")
    else:
        print(f"Failed to parse {input_xml_file}. No JSON output generated.")

if __name__ == "__main__":
    main()