import xml.etree.ElementTree as ET
import json
import os

# === Config ===
input_file = "V3.5-V2__20250527_112404.Xml"  # Adjust if needed
bit_depth = 8
adc_max_value = 2 ** bit_depth
voltage_range_mv = 10
adc_resolution = voltage_range_mv / adc_max_value

# === Parse XML ===
tree = ET.parse(input_file)
root = tree.getroot()

ns = {"ns": "urn:ge:sapphire:sapphire_3"}
wave_node = root.find(".//ns:ecgWaveform", ns)

if wave_node is None or not wave_node.attrib.get("V"):
    print("❌ ECG waveform not found.")
    exit()

# === Decode waveform ===
wave_str = wave_node.attrib["V"]
wave_samples = [int(x) for x in wave_str.strip().split()]
lead_I_data = [x * adc_resolution for x in wave_samples]

# === Get sample rate ===
sample_rate_node = root.find(".//ns:sampleRate", ns)
sampling_rate = int(sample_rate_node.attrib["V"]) if sample_rate_node is not None else 500
total_samples = len(lead_I_data)
duration_seconds = total_samples / sampling_rate

# === Build output JSON ===
json_data = {
    "metadata": {
        "sampling_rate": sampling_rate,
        "total_samples": total_samples,
        "duration_seconds": duration_seconds,
        "num_leads": 1,
        "lead_names": ["I"],
        "measured_leads": ["I"],
        "derived_leads": [],
        "units": "millivolts (mV)",
        "conversion_params": {
            "bit_depth": bit_depth,
            "adc_max_value": adc_max_value,
            "voltage_range_mv": voltage_range_mv,
            "adc_resolution": adc_resolution,
            "conversion_formula": "raw_value * adc_resolution"
        },
        "source_file": input_file,
        "data_format": "GE XML format (ecgWaveform)"
    },
    "leads": {
        "I": lead_I_data
    }
}

# === Save output ===
base_name = os.path.splitext(os.path.basename(input_file))[0]
output_file = f"XML_direct_bipolar_{base_name}.json"

with open(output_file, "w") as f:
    json.dump(json_data, f, indent=2)

print(f"✅ ECG waveform saved to {output_file}")

