import json
import matplotlib.pyplot as plt
import argparse
import os
import webbrowser
from pathlib import Path

def plot_ecg_simple(json_path, output_path):
    """Simple matplotlib plot without aspect ratio constraints."""
    with open(json_path, 'r') as f:
        ecg_data = json.load(f)

    leads = ecg_data.get("leads", {})
    metadata = ecg_data.get("metadata", {})

    if not leads:
        print("No leads to plot.")
        return

    sampling_rate = metadata.get("sampling_rate", 1000)
    lead_names = metadata.get("lead_names", sorted(leads.keys()))

    print(f"Plotting {len(lead_names)} leads from {json_path}...")

    num_leads = len(lead_names)
    fig, axes = plt.subplots(num_leads, 1, figsize=(12, 2 * num_leads), sharex=True)

    if num_leads == 1:
        axes = [axes]

    for i, lead in enumerate(lead_names):
        signal = leads[lead]
        time = [j / sampling_rate for j in range(len(signal))]
        axes[i].plot(time, signal, linewidth=0.8)
        axes[i].set_ylabel(f"{lead}\n(mV)")
        axes[i].grid(True)
        axes[i].set_xlim([0, time[-1]])

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"ECG Plot - {os.path.basename(json_path)}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved as {output_path}")
    plt.close()


def plot_ecg_conventional(json_path, output_path):
    """Matplotlib plot with conventional ECG scaling (1 second = 2.5mV)."""
    with open(json_path, 'r') as f:
        ecg_data = json.load(f)

    leads = ecg_data.get("leads", {})
    metadata = ecg_data.get("metadata", {})

    if not leads:
        print("No leads to plot.")
        return

    sampling_rate = metadata.get("sampling_rate", 1000)
    lead_names = metadata.get("lead_names", sorted(leads.keys()))

    print(f"Plotting {len(lead_names)} leads from {json_path} with conventional scaling...")

    num_leads = len(lead_names)
    fig, axes = plt.subplots(num_leads, 1, figsize=(15, 3 * num_leads), sharex=True)

    if num_leads == 1:
        axes = [axes]

    for i, lead in enumerate(lead_names):
        signal = leads[lead]
        time = [j / sampling_rate for j in range(len(signal))]
        axes[i].plot(time, signal, linewidth=0.8)
        axes[i].set_ylabel(f"{lead}\n(mV)")
        axes[i].grid(True)
        axes[i].set_xlim([0, time[-1]])
        
        # Set conventional ECG scaling: 2.5mV (y) = 1 second (x)
        axes[i].set_aspect(1/2.5)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"ECG Plot (Conventional) - {os.path.basename(json_path)}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved as {output_path}")
    plt.close()


def plot_ecg_plotly(json_path, output_path):
    """Interactive Plotly plot."""
    try:
        import plotly.graph_objs as go
    except ImportError:
        print("❌ Plotly not installed. Install with: pip install plotly")
        return

    with open(json_path, 'r') as f:
        ecg_data = json.load(f)

    leads = ecg_data.get("leads", {})
    metadata = ecg_data.get("metadata", {})

    if not leads:
        print("No leads found in JSON file.")
        return

    sampling_rate = metadata.get("sampling_rate", 500)
    lead_names = metadata.get("lead_names", sorted(leads.keys()))

    print(f"Plotting {len(lead_names)} leads at {sampling_rate} Hz with Plotly...")

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
    
    fig.write_html(output_path)
    print(f"✅ Interactive plot saved as {output_path}")
    
    # Open in browser
    abs_path = os.path.abspath(output_path)
    webbrowser.open('file://' + abs_path)
    print(f"🌐 Opening in browser...")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ECG data from JSON file with multiple plotting options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py data.json                    # Simple plot (default)
  python script.py data.json -m conventional    # Conventional ECG scaling
  python script.py data.json -m plotly          # Interactive Plotly plot
  python script.py data.json -o custom.png      # Custom output filename
        """
    )
    
    parser.add_argument('json_file', type=str, help='Path to input JSON file')
    parser.add_argument('-m', '--method', type=str, 
                        choices=['simple', 'conventional', 'plotly'],
                        default='simple',
                        help='Plotting method (default: simple)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file path (default: input_name_method.png/html)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.json_file):
        print(f"❌ Error: File '{args.json_file}' not found.")
        return
    
    # Generate output filename if not provided
    if args.output is None:
        input_path = Path(args.json_file)
        input_stem = input_path.stem
        
        if args.method == 'plotly':
            output_path = f"{input_stem}_{args.method}.html"
        else:
            output_path = f"{input_stem}_{args.method}.png"
    else:
        output_path = args.output
    
    # Call appropriate plotting function
    if args.method == 'simple':
        plot_ecg_simple(args.json_file, output_path)
    elif args.method == 'conventional':
        plot_ecg_conventional(args.json_file, output_path)
    elif args.method == 'plotly':
        plot_ecg_plotly(args.json_file, output_path)


if __name__ == "__main__":
    main()
