import os
import json
import xml.etree.ElementTree as ET
import re
from datetime import datetime

# === Configuration ===
file_group = "hl7_aecg"
input_file = "input.xml"  # Update this path
output_file = f"{file_group}_{os.path.splitext(input_file)[0]}.json"

# HL7 namespace mapping
NAMESPACES = {
    'hl7': 'urn:hl7-org:v3',
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
}

# ECG lead code mapping from MDC codes to standard names
LEAD_CODE_MAPPING = {
    'MDC_ECG_LEAD_I': 'I',
    'MDC_ECG_LEAD_II': 'II', 
    'MDC_ECG_LEAD_III': 'III',
    'MDC_ECG_LEAD_AVR': 'aVR',
    'MDC_ECG_LEAD_AVL': 'aVL',
    'MDC_ECG_LEAD_AVF': 'aVF',
    'MDC_ECG_LEAD_V1': 'V1',
    'MDC_ECG_LEAD_V2': 'V2',
    'MDC_ECG_LEAD_V3': 'V3',
    'MDC_ECG_LEAD_V4': 'V4',
    'MDC_ECG_LEAD_V5': 'V5',
    'MDC_ECG_LEAD_V6': 'V6',
    'MDC_ECG_LEAD_V3R': 'V3R',
    'MDC_ECG_LEAD_V4R': 'V4R',
    'MDC_ECG_LEAD_V5R': 'V5R'
}

def parse_hl7_timestamp(timestamp_str):
    """Parse HL7 timestamp format (YYYYMMDDHHMMSS) to readable format."""
    if len(timestamp_str) >= 14:
        return f"{timestamp_str[:4]}-{timestamp_str[4:6]}-{timestamp_str[6:8]} {timestamp_str[8:10]}:{timestamp_str[10:12]}:{timestamp_str[12:14]}"
    return timestamp_str

def extract_sequence_data(sequence_elem):
    """Extract lead data from a sequence element."""
    # Get lead code
    code_elem = sequence_elem.find('.//hl7:code', NAMESPACES)
    if code_elem is None:
        return None, None
    
    lead_code = code_elem.get('code', '')
    lead_name = LEAD_CODE_MAPPING.get(lead_code, lead_code)
    
    # Get the SLIST_PQ value element
    value_elem = sequence_elem.find('.//hl7:value[@xsi:type="SLIST_PQ"]', NAMESPACES)
    if value_elem is None:
        return lead_name, []
    
    # Extract origin, scale, and digits
    origin_elem = value_elem.find('hl7:origin', NAMESPACES)
    scale_elem = value_elem.find('hl7:scale', NAMESPACES)
    digits_elem = value_elem.find('hl7:digits', NAMESPACES)
    
    if None in [origin_elem, scale_elem, digits_elem]:
        return lead_name, []
    
    origin = float(origin_elem.get('value', '0'))
    scale = float(scale_elem.get('value', '1'))
    origin_unit = origin_elem.get('unit', 'uV')
    scale_unit = scale_elem.get('unit', 'uV')
    
    # Parse digits
    digits_text = digits_elem.text.strip() if digits_elem.text else ""
    if not digits_text:
        return lead_name, []
    
    # Split digits and convert to numbers
    digit_values = []
    for digit_str in digits_text.split():
        try:
            digit_values.append(int(digit_str))
        except ValueError:
            continue
    
    # Convert to actual voltage values
    # Formula: actual_value = origin + (digit * scale)
    voltage_values = []
    for digit in digit_values:
        actual_value = origin + (digit * scale)
        # Convert to millivolts if necessary
        if origin_unit == 'uV' or scale_unit == 'uV':
            actual_value = actual_value / 1000.0  # Convert µV to mV
        voltage_values.append(actual_value)
    
    return lead_name, voltage_values

def extract_time_sequence(sequence_elem):
    """Extract timing information from TIME_ABSOLUTE sequence."""
    value_elem = sequence_elem.find('.//hl7:value[@xsi:type="GLIST_TS"]', NAMESPACES)
    if value_elem is None:
        return None, None
    
    head_elem = value_elem.find('hl7:head', NAMESPACES)
    increment_elem = value_elem.find('hl7:increment', NAMESPACES)
    
    if head_elem is None or increment_elem is None:
        return None, None
    
    start_time = head_elem.get('value', '')
    increment_value = float(increment_elem.get('value', '0.001'))
    increment_unit = increment_elem.get('unit', 's')
    
    return start_time, increment_value

def extract_measurements(root):
    """Extract clinical measurements from annotations."""
    measurements = {}
    
    # Find all annotation elements
    annotations = root.findall('.//hl7:annotation', NAMESPACES)
    
    for annotation in annotations:
        code_elem = annotation.find('hl7:code', NAMESPACES)
        value_elem = annotation.find('hl7:value', NAMESPACES)
        
        if code_elem is not None and value_elem is not None:
            code = code_elem.get('code', '')
            
            # Extract different types of measurements
            if code == 'MDC_ECG_HEART_RATE':
                measurements['heart_rate_bpm'] = float(value_elem.get('value', '0'))
            elif code == 'MDC_ECG_TIME_PD_PR':
                measurements['pr_interval_ms'] = float(value_elem.get('value', '0'))
            elif code == 'MDC_ECG_TIME_PD_QRS':
                measurements['qrs_duration_ms'] = float(value_elem.get('value', '0'))
            elif code == 'MDC_ECG_TIME_PD_QT':
                measurements['qt_interval_ms'] = float(value_elem.get('value', '0'))
            elif code == 'MDC_ECG_TIME_PD_QTc':
                measurements['qtc_interval_ms'] = float(value_elem.get('value', '0'))
            elif code == 'MDC_ECG_ANGLE_QRS_FRONT':
                measurements['qrs_axis_degrees'] = float(value_elem.get('value', '0'))
            elif code == 'MDC_ECG_INTERPRETATION_STATEMENT':
                measurements['interpretation'] = value_elem.text.strip() if value_elem.text else ""
    
    return measurements

def parse_hl7_aecg_xml(input_file):
    """Parse HL7 aECG XML file and extract ECG data."""
    print(f"Loading HL7 aECG data from {input_file}...")
    
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None, None, None, None, None
    
    # Extract metadata
    metadata = {}
    
    # Extract effective time
    effective_time = root.find('.//hl7:effectiveTime', NAMESPACES)
    if effective_time is not None:
        low_elem = effective_time.find('hl7:low', NAMESPACES)
        high_elem = effective_time.find('hl7:high', NAMESPACES)
        if low_elem is not None:
            metadata['start_time'] = parse_hl7_timestamp(low_elem.get('value', ''))
        if high_elem is not None:
            metadata['end_time'] = parse_hl7_timestamp(high_elem.get('value', ''))
    
    # Extract device information
    device_elem = root.find('.//hl7:manufacturedSeriesDevice', NAMESPACES)
    if device_elem is not None:
        model_elem = device_elem.find('hl7:manufacturerModelName', NAMESPACES)
        if model_elem is not None and model_elem.text:
            metadata['device_model'] = model_elem.text.strip()
    
    # Extract patient demographics
    patient_elem = root.find('.//hl7:subjectDemographicPerson', NAMESPACES)
    if patient_elem is not None:
        gender_elem = patient_elem.find('.//hl7:administrativeGenderCode', NAMESPACES)
        if gender_elem is not None:
            metadata['patient_gender'] = gender_elem.get('displayName', gender_elem.get('code', ''))
    
    # Find all sequence sets (rhythm and representative beat data)
    sequence_sets = root.findall('.//hl7:sequenceSet', NAMESPACES)
    
    all_leads_data = {}
    sampling_rate = None
    total_samples = 0
    
    for seq_set in sequence_sets:
        sequences = seq_set.findall('hl7:component/hl7:sequence', NAMESPACES)
        
        # Extract timing information
        time_sequence = None
        for seq in sequences:
            code_elem = seq.find('hl7:code', NAMESPACES)
            if code_elem is not None and code_elem.get('code') == 'TIME_ABSOLUTE':
                start_time, increment = extract_time_sequence(seq)
                if increment is not None:
                    sampling_rate = 1.0 / increment  # Convert increment to sampling rate
                time_sequence = seq
                break
        
        # Extract lead data
        for seq in sequences:
            code_elem = seq.find('hl7:code', NAMESPACES)
            if code_elem is not None and code_elem.get('code') != 'TIME_ABSOLUTE':
                lead_name, lead_data = extract_sequence_data(seq)
                if lead_name and lead_data:
                    if lead_name not in all_leads_data:
                        all_leads_data[lead_name] = lead_data
                        total_samples = max(total_samples, len(lead_data))
                    else:
                        # If we have multiple datasets (rhythm + representative), use the longer one
                        if len(lead_data) > len(all_leads_data[lead_name]):
                            all_leads_data[lead_name] = lead_data
                            total_samples = max(total_samples, len(lead_data))
    
    # Extract clinical measurements
    measurements = extract_measurements(root)
    
    # Set default sampling rate if not found
    if sampling_rate is None:
        sampling_rate = 1000  # Default 1000 Hz
        print("Warning: Could not extract sampling rate from XML. Using default 1000 Hz.")
    
    print(f"Extracted {len(all_leads_data)} leads")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Total samples: {total_samples}")
    
    return all_leads_data, sampling_rate, total_samples, metadata, measurements

def main():
    # === Load and process ECG data ===
    leads, sampling_rate, num_samples_total, metadata, measurements = parse_hl7_aecg_xml(input_file)
    
    if leads is None:
        print("Failed to parse XML file.")
        return
    
    # Calculate duration
    duration_seconds = num_samples_total / sampling_rate if sampling_rate > 0 else 0
    
    print(f"Duration: {duration_seconds:.2f} seconds")
    
    # Categorize leads
    standard_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    measured_leads = []
    derived_leads = []
    additional_leads = []
    
    for lead_name in leads.keys():
        if lead_name in standard_leads:
            if lead_name in ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
                measured_leads.append(lead_name)
            else:
                derived_leads.append(lead_name)
        else:
            additional_leads.append(lead_name)
    
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
            "additional_leads": additional_leads,
            "units": "millivolts (mV)",
            "source_file": input_file,
            "data_format": "HL7 aECG XML format",
            "conversion_info": {
                "original_units": "microvolts (µV)",
                "conversion_factor": 0.001,
                "conversion_formula": "(origin + digit * scale) / 1000"
            }
        },
        "clinical_data": measurements,
        "device_info": {},
        "leads": leads
    }
    
    # Add metadata to appropriate sections
    if metadata:
        complete_data["device_info"].update(metadata)
    
    # === Save to JSON file ===
    print(f"Saving to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(complete_data, f, indent=2)
    
    print(f"Successfully saved complete ECG data to '{output_file}'")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    print(f"Leads included: {', '.join(leads.keys())}")
    
    # === Show data range summary ===
    print(f"\nData range summary:")
    
    def show_lead_ranges(lead_list, category_name):
        if not lead_list:
            return
        print(f"  {category_name}:")
        for lead_name in lead_list:
            if lead_name in leads and len(leads[lead_name]) > 0:
                lead_data = leads[lead_name]
                min_val = min(lead_data)
                max_val = max(lead_data)
                mean_val = sum(lead_data) / len(lead_data)
                print(f"    {lead_name}: [{min_val:.3f}, {max_val:.3f}] mV (mean: {mean_val:.3f})")
    
    show_lead_ranges(measured_leads, "Measured leads")
    show_lead_ranges(derived_leads, "Derived leads") 
    show_lead_ranges(additional_leads, "Additional leads")
    
    # === Clinical measurements summary ===
    if measurements:
        print(f"\nClinical measurements:")
        for key, value in measurements.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
    
    # === Data quality checks ===
    print(f"\nData quality checks:")
    
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
    
    # Check lead length consistency
    lead_lengths = {lead: len(data) for lead, data in leads.items()}
    unique_lengths = set(lead_lengths.values())
    
    if len(unique_lengths) > 1:
        print("  Warning: Leads have different lengths!")
        for lead, length in lead_lengths.items():
            print(f"    {lead}: {length} samples")
    else:
        print("  ✓ All leads have consistent length")

if __name__ == "__main__":
    main()
