import os
import json
import numpy as np
from pydicom import dcmread
from pathlib import Path

# === Configuration ===
input_directory = "RBH"  # Directory with input DICOM files
output_directory = "output_dicom"  # Where JSON files will be saved
file_group = "DICOM_RBH"
SAMPLING_RATE = 500  # Hz
ADC_RESOLUTION_DIVISOR = 2500 / 1.25  # For scaling microvolt to millivolt

def create_output_filename(file_path, input_dir, file_group):
    rel_path = os.path.relpath(file_path, input_dir)
    dir_path, filename = os.path.split(rel_path)
    base_name = Path(filename).stem
    if dir_path and dir_path != '.':
        dir_str = dir_path.replace('/', '_').replace('\\', '_')
        output_name = f"{file_group}_{dir_str}_{base_name}.json"
    else:
        output_name = f"{file_group}_{base_name}.json"
    return output_name

def parse_dicom_ecg_file(file_path):
    try:
        ds = dcmread(file_path)
    except Exception as e:
        print(f"  ❌ Failed to read DICOM: {e}")
        return None

    ecg_data = {
        "metadata": {
            "source_file": file_path,
            "data_format": "DICOM SCP-ECG",
            "sampling_rate": SAMPLING_RATE,
        },
        "leads": {}
    }

    try:
        waveform = ds.waveform_array(0)  # Shape: (num_samples, num_leads)
        num_samples, num_leads = waveform.shape
    except Exception as e:
        print(f"  ❌ Failed to extract waveform: {e}")
        return None

    # Extract scale if present
    try:
        # Microvolt scale per lead (assumed same for all if not per-lead)
        scales = [float(wf_channel.MultiplexGroup.Sensitivity) for wf_channel in ds.WaveformSequence[0].ChannelDefinitionSequence]
    except Exception:
        scales = [1.0] * num_leads  # fallback

    # Convert waveform to mV
    print(scales)
    leads = {}
    standard_lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                       'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    for i in range(num_leads):
        if i < len(standard_lead_names):
            # lead_name = ds.WaveformSequence[0].ChannelDefinitionSequence[i].ChannelSourceCodeSequence[0].CodeValue
            lead_name = standard_lead_names[i]
        else:
            lead_name = f"Extra_{i+1}"

        raw_signal = waveform[:, i]
        scale = scales[i]
        mv_signal = (raw_signal * scale) / ADC_RESOLUTION_DIVISOR
        leads[lead_name] = mv_signal.tolist()

    ecg_data["leads"] = leads
    ecg_data["metadata"]["num_leads"] = num_leads
    ecg_data["metadata"]["lead_names"] = list(leads.keys())
    ecg_data["metadata"]["units"] = "millivolts (mV)"
    ecg_data["metadata"]["total_samples"] = num_samples
    ecg_data["metadata"]["duration_seconds"] = num_samples / SAMPLING_RATE
    ecg_data["metadata"]["conversion_params"] = {
        "adc_resolution_divisor": ADC_RESOLUTION_DIVISOR,
        "sampling_rate": SAMPLING_RATE,
        "conversion_formula": "converted_value = (raw_value * scale) / 2500",
        "raw_data_units": "ADC counts",
        "intermediate_units": "microvolts (uV) after scale multiplication",
        "final_output_units": "millivolts (mV)"
    }

    return ecg_data

def process_dicom_file(file_path, output_dir, input_dir, file_group):
    print(f"Processing: {file_path}")
    parsed_data = parse_dicom_ecg_file(file_path)
    if not parsed_data or not parsed_data.get("leads"):
        print(f"  ❌ Skipping: No leads extracted.")
        return False

    output_filename = create_output_filename(file_path, input_dir, file_group)
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:
        json.dump(parsed_data, f, indent=4)

    print(f"  ✅ Saved to {output_filename}")
    return True

def main():
    global file_group
    os.makedirs(output_directory, exist_ok=True)

    use_default = input(f"Use default file group prefix '{file_group}'? (y/n): ").lower().strip()
    if use_default != 'y':
        file_group = input("Enter file group prefix: ").strip() or file_group

    dicom_files = []
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))

    if not dicom_files:
        print("❌ No DICOM files found.")
        return

    print(f"📁 Found {len(dicom_files)} DICOM files")
    print(f"📤 Output directory: {output_directory}")
    print(f"🏷️  File group prefix: {file_group}")
    print(f"⚙️  Sampling rate: {SAMPLING_RATE} Hz")
    print(f"🔧 ADC resolution: scale/{ADC_RESOLUTION_DIVISOR}")
    print("-" * 50)

    successful = 0
    for file_path in dicom_files:
        if process_dicom_file(file_path, output_directory, input_directory, file_group):
            successful += 1

    print("-" * 50)
    print(f"🎉 Done! {successful} files processed.")
    print(f"📁 Output saved in: {output_directory}")

if __name__ == "__main__":
    main()

