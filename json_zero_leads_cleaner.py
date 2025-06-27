import json
import os
from pathlib import Path

# === Configuration ===
input_directory = "converted_json"  # Directory containing JSON files
output_directory = "cleaned"  # Directory for cleaned JSON files (optional - set to None to overwrite originals)
backup_directory = "backup"  # Directory to backup original files (set to None to skip backup)
create_report = True  # Whether to create a cleaning report

def is_lead_all_zeros(lead_data, tolerance=1e-10):
    """
    Check if a lead contains only zero values (with small tolerance for floating point errors)
    """
    if not isinstance(lead_data, list):
        return False
    
    if len(lead_data) == 0:
        return True
    
    return all(abs(value) <= tolerance for value in lead_data)

def clean_json_file(file_path, output_dir=None, backup_dir=None):
    """
    Clean a single JSON file by removing leads with only zero values
    Returns: (success, original_leads_count, cleaned_leads_count, removed_leads)
    """
    try:
        # Load JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if the file has the expected structure
        if 'leads' not in data or not isinstance(data['leads'], dict):
            print(f"  ⚠️  File {file_path} doesn't have expected 'leads' structure. Skipping...")
            return False, 0, 0, []
        
        original_leads = data['leads'].copy()
        original_count = len(original_leads)
        removed_leads = []
        
        # Check each lead for zero values
        for lead_name, lead_data in list(original_leads.items()):
            if is_lead_all_zeros(lead_data):
                removed_leads.append(lead_name)
                del data['leads'][lead_name]
        
        cleaned_count = len(data['leads'])
        
        # If no leads were removed, skip processing
        if not removed_leads:
            print(f"  ✅ No zero leads found in {os.path.basename(file_path)}")
            return True, original_count, cleaned_count, removed_leads
        
        # Create backup if requested
        if backup_dir:
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, os.path.basename(file_path))
            with open(backup_path, 'w') as f:
                json.dump({**data, 'leads': original_leads}, f, indent=2)
        
        # Update metadata
        if 'metadata' in data:
            data['metadata']['num_leads'] = cleaned_count
            data['metadata']['lead_names'] = sorted(list(data['leads'].keys()))
            
            # Add cleaning info to metadata
            if 'processing_history' not in data['metadata']:
                data['metadata']['processing_history'] = []
            
            data['metadata']['processing_history'].append({
                'operation': 'zero_leads_removal',
                'removed_leads': removed_leads,
                'original_lead_count': original_count,
                'final_lead_count': cleaned_count
            })
        
        # Determine output path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(file_path))
        else:
            output_path = file_path  # Overwrite original
        
        # Save cleaned file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  ✅ Cleaned {os.path.basename(file_path)}: {original_count} → {cleaned_count} leads (removed: {', '.join(removed_leads)})")
        return True, original_count, cleaned_count, removed_leads
        
    except Exception as e:
        print(f"  ❌ Error processing {file_path}: {str(e)}")
        return False, 0, 0, []

def scan_json_files(directory):
    """Recursively find all JSON files in directory"""
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def create_cleaning_report(results, report_path):
    """Create a detailed cleaning report"""
    try:
        total_files = len(results)
        successful = sum(1 for r in results if r['success'])
        failed = total_files - successful
        total_original_leads = sum(r['original_leads'] for r in results if r['success'])
        total_final_leads = sum(r['final_leads'] for r in results if r['success'])
        total_removed_leads = total_original_leads - total_final_leads
        
        report = {
            "summary": {
                "total_files_processed": total_files,
                "successful": successful,
                "failed": failed,
                "total_original_leads": total_original_leads,
                "total_final_leads": total_final_leads,
                "total_removed_leads": total_removed_leads
            },
            "files": results,
            "removed_leads_by_name": {}
        }
        
        # Count which lead names were removed most often
        lead_removal_count = {}
        for r in results:
            if r['success']:
                for lead in r['removed_leads']:
                    lead_removal_count[lead] = lead_removal_count.get(lead, 0) + 1
        
        report["removed_leads_by_name"] = dict(sorted(lead_removal_count.items(), key=lambda x: x[1], reverse=True))
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Cleaning report saved to: {report_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating report: {e}")
        return False

def main():
    print("🧹 JSON Zero Leads Cleaner")
    print("=" * 50)
    
    # Show configuration
    print(f"📁 Input directory: {input_directory}")
    if output_directory:
        print(f"📤 Output directory: {output_directory}")
    else:
        print("📤 Output: Overwriting original files")
    
    if backup_directory:
        print(f"💾 Backup directory: {backup_directory}")
    else:
        print("💾 Backup: Disabled")
    
    # Confirm settings
    proceed = input("\nProceed with these settings? (y/n): ").lower().strip()
    if proceed != 'y':
        print("Operation cancelled.")
        return
    
    # Find all JSON files
    json_files = scan_json_files(input_directory)
    
    if not json_files:
        print("❌ No JSON files found in the specified directory.")
        return
    
    print(f"\n📁 Found {len(json_files)} JSON files to process...")
    print("-" * 50)
    
    # Process each file
    results = []
    
    for file_path in json_files:
        print(f"Processing: {os.path.relpath(file_path, input_directory)}")
        
        success, original_count, final_count, removed_leads = clean_json_file(
            file_path, output_directory, backup_directory
        )
        
        results.append({
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "success": success,
            "original_leads": original_count,
            "final_leads": final_count,
            "removed_leads": removed_leads
        })
    
    # Summary
    print("-" * 50)
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    total_removed = sum(len(r['removed_leads']) for r in results if r['success'])
    
    print(f"🎉 Processing complete!")
    print(f"✅ Successfully processed: {successful} files")
    print(f"❌ Failed: {failed} files")
    print(f"🗑️  Total leads removed: {total_removed}")
    
    # Show most commonly removed leads
    if total_removed > 0:
        lead_counts = {}
        for r in results:
            if r['success']:
                for lead in r['removed_leads']:
                    lead_counts[lead] = lead_counts.get(lead, 0) + 1
        
        if lead_counts:
            print(f"\n🏷️  Most commonly removed leads:")
            for lead, count in sorted(lead_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {lead}: {count} files")
    
    # Create report if requested
    if create_report:
        report_path = os.path.join(output_directory or input_directory, "cleaning_report.json")
        create_cleaning_report(results, report_path)
    
    print(f"\n📁 Output location: {output_directory or input_directory}")

if __name__ == "__main__":
    main()
