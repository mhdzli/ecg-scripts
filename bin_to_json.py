import numpy as np
import json
import os

# === Configuration ===
file_group = "holter"
input_file = "input.bin"
output_file = f"{file_group}_{os.path.splitext(input_file)[0]}.json"
num_leads = 8
raw_lead_names = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
sampling_rate = 1000  # Hz
scale_factor = 2500

# === Load binary and reshape ===
print("Loading binary data...")
raw_data = np.fromfile(input_file, dtype=np.int16)
ecg_data = raw_data.reshape(-1, num_leads)
num_samples_total = ecg_data.shape[0]

print(f"Total samples: {num_samples_total}")
print(f"Duration: {num_samples_total / sampling_rate:.2f} seconds")

# === Extract and scale all leads ===
print("Processing leads...")
leads = {}

# Scale raw leads
for i, lead in enumerate(raw_lead_names):
    leads[lead] = (ecg_data[:, i] / scale_factor).tolist()

# Calculate derived leads
print("Calculating derived leads...")
leads["III"] = [leads["II"][i] - leads["I"][i] for i in range(num_samples_total)]
leads["aVR"] = [-(leads["I"][i] + leads["II"][i]) / 2 for i in range(num_samples_total)]
leads["aVL"] = [leads["I"][i] - leads["II"][i] / 2 for i in range(num_samples_total)]
leads["aVF"] = [leads["II"][i] - leads["I"][i] / 2 for i in range(num_samples_total)]

# === Create complete data structure ===
complete_data = {
    "metadata": {
        "sampling_rate": sampling_rate,
        "total_samples": num_samples_total,
        "duration_seconds": num_samples_total / sampling_rate,
        "num_leads": len(leads),
        "lead_names": list(leads.keys()),
        "scale_factor_applied": scale_factor,
        "source_file": input_file
    },
    "leads": leads
}

# === Save to JSON file ===
print(f"Saving to {output_file}...")
with open(output_file, "w") as f:
    json.dump(complete_data, f, indent=2)

print(f"Successfully saved complete ECG data to '{output_file}'")
print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
print(f"Leads included: {', '.join(leads.keys())}")
