import json
import os
import pyedflib
import numpy as np

# === Configuration ===
input_file = "04-32-13.EDF"
output_file = f"edf_{os.path.splitext(input_file)[0]}.json"

bit_depth = 16
adc_max_value = 2 ** bit_depth
voltage_range_mv = 10
adc_resolution = voltage_range_mv / adc_max_value

measured_leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
derived_leads = ['III', 'aVR', 'aVL', 'aVF']

def edf_to_json(input_file):
    f = pyedflib.EdfReader(input_file)
    n_signals = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sampling_rate = f.getSampleFrequency(0)
    num_samples = f.getNSamples()[0]
    
    lead_data = {}
    
    for i in range(n_signals):
        label = signal_labels[i]
        signal = f.readSignal(i)
        
        # Convert to mV if needed
        mv_signal = signal * adc_resolution
        lead_data[label] = mv_signal.tolist()

    f._close()

    duration_seconds = num_samples / sampling_rate
    
    complete_data = {
        "metadata": {
            "sampling_rate": sampling_rate,
            "total_samples": num_samples,
            "duration_seconds": duration_seconds,
            "num_leads": len(lead_data),
            "lead_names": list(lead_data.keys()),
            "measured_leads": measured_leads,
            "derived_leads": derived_leads,
            "units": "millivolts (mV)",
            "conversion_params": {
                "bit_depth": bit_depth,
                "adc_max_value": adc_max_value,
                "voltage_range_mv": voltage_range_mv,
                "adc_resolution": adc_resolution,
                "conversion_formula": "raw_value * adc_resolution"
            },
            "source_file": input_file,
            "data_format": "EDF format"
        },
        "leads": lead_data
    }

    with open(output_file, "w") as f_json:
        json.dump(complete_data, f_json, indent=2)

    print(f"Saved JSON to {output_file}")

if __name__ == "__main__":
    edf_to_json(input_file)
