import pyedflib
import json
import os

# === Configuration ===
input_file = "input.EDF"  # change this if needed

# === ADC conversion parameters (assume 16-bit, ±5mV range) ===
bit_depth = 16
adc_max_value = 2 ** bit_depth
voltage_range_mv = 10  # ±5mV = 10 mV total
adc_resolution = voltage_range_mv / adc_max_value * 2.5

measured_leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
derived_leads = ['III', 'aVR', 'aVL', 'aVF']
all_lead_names = measured_leads + derived_leads

# === Open EDF ===
edf = pyedflib.EdfReader(input_file)
n_signals = edf.signals_in_file
labels = edf.getSignalLabels()
sampling_rates = [edf.getSampleFrequency(i) for i in range(n_signals)]
samples_counts = edf.getNSamples()

# === Duration check ===
max_duration = max(ns / sr for ns, sr in zip(samples_counts, sampling_rates))
print(f"🕒 Total duration: {max_duration:.2f} seconds")

# === Prompt user for time range ===
segment_start = float(input("Enter segment start time (seconds): "))
segment_end = float(input("Enter segment end time (seconds): "))

if segment_end > max_duration or segment_start >= segment_end:
    print("❌ Invalid segment range.")
    edf._close()
    exit()

# === Find ECG channel (we assume it's the first or named "ECG") ===
ecg_index = None
for i, label in enumerate(labels):
    if label.upper() in ["ECG", "I"]:
        ecg_index = i
        break

if ecg_index is None:
    print("❌ ECG lead (I or ECG) not found.")
    edf._close()
    exit()

sampling_rate = sampling_rates[ecg_index]
start_sample = int(segment_start * sampling_rate)
end_sample = int(segment_end * sampling_rate)
total_samples = end_sample - start_sample

raw_segment = edf.readSignal(ecg_index)[start_sample:end_sample]
edf._close()

# Convert ADC to mV
lead_I_data = [val * adc_resolution for val in raw_segment]

# === Build output dictionary ===
complete_data = {
    "metadata": {
        "sampling_rate": sampling_rate,
        "total_samples": total_samples,
        "duration_seconds": total_samples / sampling_rate,
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
        "data_format": "EDF format (extracted segment)"
    },
    "leads": {
        "I": lead_I_data
    }
}

# === Save file ===
base_name = os.path.splitext(os.path.basename(input_file))[0]
output_file = f"EDF_{base_name}_SS_{int(segment_start)}_TT_{int(segment_end)}.json"

with open(output_file, "w") as f:
    json.dump(complete_data, f, indent=2)

print(f"\n✅ Saved lead I segment to {output_file}")

