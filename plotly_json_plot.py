import json
import plotly.graph_objs as go
import os

def plot_ecg_from_json(json_path):
    # === Load the JSON file ===
    with open(json_path, 'r') as f:
        ecg_data = json.load(f)

    leads = ecg_data.get("leads", {})
    metadata = ecg_data.get("metadata", {})

    if not leads:
        print("No leads found in JSON file.")
        return

    sampling_rate = metadata.get("sampling_rate", 500)  # default to 500 Hz
    duration = metadata.get("duration_seconds", None)
    lead_names = metadata.get("lead_names", sorted(leads.keys()))

    print(f"Plotting {len(lead_names)} leads at {sampling_rate} Hz...")

    fig = go.Figure()

    for idx, lead in enumerate(lead_names):
        signal = leads[lead]
        time_axis = [i / sampling_rate for i in range(len(signal))]
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=[s + idx * 2 for s in signal],  # vertical offset for visibility
            mode='lines',
            name=lead
        ))

    fig.update_layout(
        title=f"ECG Plot - {os.path.basename(json_path)}",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (offset mV)",
        legend=dict(title="Leads"),
        height=600,
        template='plotly_white'
    )
    fig.show()

# === Configuration ===
json_file = "input.json"  # Replace with your actual output JSON path
plot_ecg_from_json(json_file)
