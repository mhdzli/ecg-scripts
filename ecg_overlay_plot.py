"""
ECG Overlay Comparison Plot
Overlays EDF and Holter ECG signals for visual comparison at matched time segments.

Usage:
    python script.py --edf path/to/edf.json --holter path/to/holter.json --edf-start 10.5 --holter-start 0.0 --holter-end 30.0
    python script.py -e edf.json -H holter.json --edf-start 10.5 --holter-start 0 --holter-end 30 -o comparison.png
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

def load_ecg_json(json_path):
    """Load ECG data from JSON file"""
    try:
        with open(json_path, 'r') as f:
            ecg_data = json.load(f)
        return ecg_data
    except FileNotFoundError:
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file: {json_path}")
        sys.exit(1)

def extract_time_segment(signal, sampling_rate, start_time, end_time):
    """
    Extract a time segment from a signal
    
    Args:
        signal: List of signal values
        sampling_rate: Sampling rate in Hz
        start_time: Start time in seconds
        end_time: End time in seconds (None for end of signal)
    
    Returns:
        Tuple of (time_array, signal_segment)
    """
    start_sample = int(start_time * sampling_rate)
    
    if end_time is None:
        end_sample = len(signal)
    else:
        end_sample = int(end_time * sampling_rate)
    
    # Ensure we don't go out of bounds
    start_sample = max(0, start_sample)
    end_sample = min(len(signal), end_sample)
    
    signal_segment = signal[start_sample:end_sample]
    
    # Create time array relative to the start time
    time_array = np.arange(len(signal_segment)) / sampling_rate
    
    return time_array, signal_segment

def plot_overlay_comparison(edf_path, holter_path, edf_start, holter_start, holter_end, output_path):
    """
    Plot EDF and Holter ECG signals overlaid for comparison
    """
    print("="*70)
    print("ECG OVERLAY COMPARISON")
    print("="*70)
    
    # Load both JSON files
    print(f"\nLoading EDF file: {edf_path}")
    edf_data = load_ecg_json(edf_path)
    
    print(f"Loading Holter file: {holter_path}")
    holter_data = load_ecg_json(holter_path)
    
    # Extract metadata
    edf_leads = edf_data.get("leads", {})
    holter_leads = holter_data.get("leads", {})
    
    edf_metadata = edf_data.get("metadata", {})
    holter_metadata = holter_data.get("metadata", {})
    
    edf_sampling_rate = edf_metadata.get("sampling_rate", 250)
    holter_sampling_rate = holter_metadata.get("sampling_rate", 1000)
    
    edf_lead_names = edf_metadata.get("lead_names", sorted(edf_leads.keys()))
    holter_lead_names = holter_metadata.get("lead_names", sorted(holter_leads.keys()))
    
    print(f"\nEDF sampling rate: {edf_sampling_rate} Hz")
    print(f"Holter sampling rate: {holter_sampling_rate} Hz")
    print(f"EDF leads: {edf_lead_names}")
    print(f"Holter leads: {holter_lead_names}")
    
    # Find common leads - plot ALL Holter leads and overlay with matching EDF leads
    leads_to_plot = holter_lead_names.copy()
    
    print(f"\nWill plot all {len(leads_to_plot)} Holter leads")
    
    # Check which leads have matches in EDF
    matched_leads = [lead for lead in leads_to_plot if lead in edf_lead_names]
    if matched_leads:
        print(f"Leads with exact name match in EDF: {matched_leads}")
    
    unmatched_leads = [lead for lead in leads_to_plot if lead not in edf_lead_names]
    if unmatched_leads:
        print(f"Leads without exact match (will try positional matching): {unmatched_leads}")
    
    # Calculate duration
    holter_duration = holter_end - holter_start
    edf_end = edf_start + holter_duration
    
    print(f"\nTime segments:")
    print(f"  Holter: {holter_start:.3f}s to {holter_end:.3f}s (duration: {holter_duration:.3f}s)")
    print(f"  EDF: {edf_start:.3f}s to {edf_end:.3f}s (duration: {holter_duration:.3f}s)")
    
    # Create figure - plot ALL Holter leads
    num_leads = len(leads_to_plot)
    fig, axes = plt.subplots(num_leads, 1, figsize=(15, 3 * num_leads), sharex=True)
    
    if num_leads == 1:
        axes = [axes]
    
    # Plot each lead
    for i, lead in enumerate(leads_to_plot):
        # Extract Holter segment (always present since we're using holter_lead_names)
        holter_signal = holter_leads[lead]
        holter_time, holter_segment = extract_time_segment(
            holter_signal, holter_sampling_rate, holter_start, holter_end
        )
        
        # Plot Holter signal (in red)
        axes[i].plot(holter_time, holter_segment, 
                    linewidth=0.8, color='red', label='Holter', alpha=0.7)
        
        # Try to find matching EDF lead
        edf_lead_name = None
        edf_plotted = False
        
        # First try exact name match
        if lead in edf_leads:
            edf_lead_name = lead
        # If no exact match, try positional matching
        elif i < len(edf_lead_names):
            edf_lead_name = edf_lead_names[i]
            print(f"  Plotting: Holter '{lead}' with EDF '{edf_lead_name}' (positional match)")
        
        # Plot EDF signal if we found a match
        if edf_lead_name and edf_lead_name in edf_leads:
            edf_signal = edf_leads[edf_lead_name]
            edf_time, edf_segment = extract_time_segment(
                edf_signal, edf_sampling_rate, edf_start, edf_end
            )
            
            # Plot EDF signal (in blue)
            axes[i].plot(edf_time, edf_segment, 
                        linewidth=0.8, color='blue', label=f'EDF ({edf_lead_name})', alpha=0.7)
            edf_plotted = True
            
            # Include both signals in y-axis calculation
            all_values = list(holter_segment) + list(edf_segment)
        else:
            print(f"  Warning: No EDF lead available for Holter lead '{lead}'")
            all_values = list(holter_segment)
        
        # Configure subplot
        lead_label = f"{lead}"
        if edf_plotted and edf_lead_name != lead:
            lead_label += f" / {edf_lead_name}"
        
        axes[i].set_ylabel(f"{lead_label}\n(mV)", fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right', fontsize=8)
        
        # Set conventional ECG scaling: 2.5mV (y) = 1 second (x)
        axes[i].set_aspect(1/2.5)
        
        # Set y-axis limits based on data range with some padding
        if all_values:
            y_min, y_max = min(all_values), max(all_values)
            y_range = y_max - y_min
            axes[i].set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # Set x-axis label and limits
    axes[-1].set_xlabel("Time (s)", fontsize=11)
    axes[-1].set_xlim([0, holter_duration])
    
    # Add title with matching information
    title = f"ECG Overlay Comparison - All {num_leads} Leads\n"
    title += f"EDF: {os.path.basename(edf_path)} (t={edf_start:.2f}s-{edf_end:.2f}s) | "
    title += f"Holter: {os.path.basename(holter_path)} (t={holter_start:.2f}s-{holter_end:.2f}s)"
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Overlay plot saved as '{output_path}'")
    
    # Also show the plot
    plt.show()
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(
        description='Overlay EDF and Holter ECG signals for visual comparison.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python script.py --edf edf.json --holter holter.json --edf-start 10.5 --holter-start 0 --holter-end 30

  # With custom output
  python script.py -e edf.json -H holter.json --edf-start 15.2 --holter-start 0 --holter-end 60 -o comparison.png
        """
    )
    
    parser.add_argument('--edf', '-e', required=True, 
                       help='Path to EDF JSON file')
    parser.add_argument('--holter', '-H', required=True, 
                       help='Path to Holter JSON file')
    parser.add_argument('--edf-start', type=float, required=True, 
                       help='Start time in EDF recording (seconds) where match begins')
    parser.add_argument('--holter-start', type=float, required=True, 
                       help='Start time in Holter recording (seconds)')
    parser.add_argument('--holter-end', type=float, required=True, 
                       help='End time in Holter recording (seconds)')
    parser.add_argument('--output', '-o', default='ecg_overlay_comparison.png', 
                       help='Output plot filename (default: ecg_overlay_comparison.png)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.holter_end <= args.holter_start:
        print("Error: holter-end must be greater than holter-start")
        sys.exit(1)
    
    if args.edf_start < 0 or args.holter_start < 0:
        print("Error: Start times cannot be negative")
        sys.exit(1)
    
    # Run the overlay plot
    plot_overlay_comparison(
        args.edf, 
        args.holter, 
        args.edf_start, 
        args.holter_start, 
        args.holter_end, 
        args.output
    )

if __name__ == "__main__":
    main()
