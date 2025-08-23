import base64
import struct
import xml.etree.ElementTree as ET
import json

# Configuration
ADC_DIVISOR = 2500  # For converting μV to mV
SAMPLING_RATE = 500  # Hz

def decode_waveform(base64_data, sample_size=2, units_per_bit=4.88):
    binary_data = base64.b64decode(base64_data)
    samples = struct.unpack('<' + 'h' * (len(binary_data) // sample_size), binary_data)
    return [(s * units_per_bit) / ADC_DIVISOR for s in samples]

def parse_muse_xml(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()

    ecg_data = {
        "metadata": {
            "source_file": filepath,
            "data_format": "GE Muse RestingECG XML",
            "sampling_rate": SAMPLING_RATE,
        },
        "leads": {}
    }

    # === Extract Demographics ===
    demo = root.find('PatientDemographics')
    if demo is not None:
        ecg_data["metadata"]["subject_id"] = demo.findtext('PatientID', '').strip()
        ecg_data["metadata"]["gender"] = demo.findtext('Gender')
        ecg_data["metadata"]["age"] = demo.findtext('PatientAge')

    # === ECG Measurements ===
    measurements = root.find('RestingECGMeasurements')
    if measurements is not None:
        annotations = {}
        for elem in measurements:
            if elem.tag and elem.text and elem.text.strip():
                try:
                    annotations[elem.tag] = float(elem.text)
                except ValueError:
                    annotations[elem.tag] = elem.text
        ecg_data["metadata"]["annotations"] = annotations

    # === Diagnosis ===
    diagnosis = root.find('Diagnosis')
    if diagnosis is not None:
        statements = diagnosis.findall('./DiagnosisStatement/StmtText')
        ecg_data["metadata"]["diagnosis"] = [stmt.text.strip() for stmt in statements if stmt.text]

    # === Waveform Data (Median or Rhythm) ===
    waveform_block = root.find('./Waveform')
    if waveform_block is not None:
        for lead in waveform_block.findall('LeadData'):
            lead_id = lead.findtext('LeadID')
            data_base64 = lead.findtext('WaveFormData')
            units_per_bit = float(lead.findtext('LeadAmplitudeUnitsPerBit', '4.88'))

            if lead_id and data_base64:
                ecg_data["leads"][lead_id] = decode_waveform(data_base64, sample_size=2, units_per_bit=units_per_bit)

    # === Add derived metadata ===
    if ecg_data["leads"]:
        first_lead = next(iter(ecg_data["leads"].values()))
        total_samples = len(first_lead)
        ecg_data["metadata"]["total_samples"] = total_samples
        ecg_data["metadata"]["duration_seconds"] = total_samples / SAMPLING_RATE
        ecg_data["metadata"]["num_leads"] = len(ecg_data["leads"])
        ecg_data["metadata"]["lead_names"] = sorted(list(ecg_data["leads"].keys()))
        ecg_data["metadata"]["units"] = "millivolts (mV)"
        ecg_data["metadata"]["conversion_params"] = {
            "adc_resolution_divisor": ADC_DIVISOR,
            "sampling_rate": SAMPLING_RATE,
            "conversion_formula": "mV = (raw_value * μV_per_bit) / 2500",
            "raw_data_units": "ADC counts",
            "intermediate_units": "microvolts (μV)",
            "final_output_units": "millivolts (mV)"
        }

    return ecg_data

# === Example usage ===
if __name__ == "__main__":
    input_file = "RASEMA0011_ECG_2023_anonymised.xml"
    output_file = "RASEMA0011_ECG_2023_converted.json"
    
    parsed = parse_muse_xml(input_file)
    
    with open(output_file, 'w') as f:
        json.dump(parsed, f, indent=4)

    print(f"✅ Converted and saved to {output_file}")

