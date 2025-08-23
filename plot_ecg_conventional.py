import json
import matplotlib.pyplot as plt
import os

def plot_ecg_matplotlib(json_path):
    # === Load the JSON file ===
    with open(json_path, 'r') as f:
        ecg_data = json.load(f)

    leads = ecg_data.get("leads", {})
    metadata = ecg_data.get("metadata", {})

    if not leads:
        print("No leads to plot.")
        return

    sampling_rate = metadata.get("sampling_rate", 1000)  # default to 1000 Hz
    lead_names = metadata.get("lead_names", sorted(leads.keys()))

    print(f"Plotting {len(lead_names)} leads from {json_path}...")

    num_leads = len(lead_names)
    fig, axes = plt.subplots(num_leads, 1, figsize=(15, 3 * num_leads), sharex=True)

    if num_leads == 1:
        axes = [axes]  # make iterable

    for i, lead in enumerate(lead_names):
        signal = leads[lead]
        time = [j / sampling_rate for j in range(len(signal))]
        axes[i].plot(time, signal, linewidth=0.8)
        axes[i].set_ylabel(f"{lead}\n(mV)")
        axes[i].grid(True)
        axes[i].set_xlim([0, time[-1]])
        
        # Set conventional ECG scaling: 2.5mV (y) = 1 second (x)
        axes[i].set_aspect(1/2.5)  # 1 second per 2.5 mV

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"ECG Plot - {os.path.basename(json_path)}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()
    plt.savefig("output.png", dpi=300, bbox_inches='tight')
    print("✅ Plot saved as output.png")


# === Configuration ===
json_file = "input.json"  # Replace with your actual JSON file
plot_ecg_matplotlib(json_file)
