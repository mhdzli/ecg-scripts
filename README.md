# ECG Data JSON Conversion Scripts

## China DAT to Single JSON Converter

### Key Features:

### **Consistent JSON Structure**

- **Metadata section** with comprehensive information about the data
- **Leads section** containing all ECG lead data
- Same organizational pattern as the Holter converter

### **Enhanced Metadata**

- **Conversion parameters** - documents the ADC-to-millivolt conversion
- **Lead categorization** - separates measured vs derived leads
- **Units specification** - clearly indicates data is in millivolts
- **Source tracking** - records original file path and format type

### **Improved Processing**

- **Better error handling** for text vs binary file loading
- **Progress feedback** during processing
- **Data validation** with range summaries
- **Comprehensive logging** of conversion steps

### **Technical Improvements**

- **Proper scaling** using ADC conversion parameters
- **Maintains precision** in millivolt conversion
- **Same derived lead calculations** as original script
- **Memory-efficient processing** for large files

## Usage:

Simply update the `input_file` variable at the top of the script and run:

```python
python china_to_json.py
```

The script will:

1. **Load** China DAT file (handles both text and binary formats)
2. **Convert** ADC values to millivolts using specified parameters
3. **Calculate** derived leads (III, aVR, aVL, aVF)
4. **Save** everything to a single JSON file with comprehensive metadata
5. **Display** processing information and data ranges

## Binary to Single JSON Converter (`bin_to_json.py`)

This script converts your binary ECG data into a single comprehensive JSON file. Key features:

- **Loads and processes** the entire binary file at once
- **Applies scaling** using the same scale factor (2500)
- **Calculates derived leads** (III, aVR, aVL, aVF) from the raw leads
- **Includes comprehensive metadata** (sampling rate, duration, lead names, etc.)
- **Saves everything** to a single JSON file with proper structure

## Bard TXT to Single JSON Converter

### Key Features:

### **Consistent JSON Structure**

- **Same metadata section** format as Holter and China converters
- **Unified leads section** containing all 12 ECG leads
- **Compatible** with the existing segmenter script

### **Enhanced Data Processing**

- **Automatic sampling rate detection** from the file header
- **Proper ADC-to-millivolt conversion** using specified parameters
- **All 12 standard ECG leads** processed (including both measured and derived)
- **Data validation** with comprehensive quality checks

### **Improved Error Handling**

- **Robust parsing** with fallback values for invalid data
- **Length verification** ensures all leads have consistent sample counts
- **Warning system** for data quality issues
- **Progress tracking** during file processing

### **Quality Assurance**

- **Data range analysis** showing min/max/mean values for each lead
- **Zero-value detection** to identify potential missing data
- **High-value detection** to catch potential parsing errors
- **Lead categorization** (measured vs derived)

## Key Improvements Over Original:

### **Better Structure**

- **Comprehensive metadata** tracking conversion parameters
- **Lead categorization** distinguishing measured from derived leads
- **Units specification** clearly indicating millivolt values

### **Enhanced Processing**

- **More robust parsing** with better error handling
- **Data quality validation** with multiple checks
- **Progress feedback** during processing
- **Statistical analysis** of the converted data

### **Consistency**

- **Same JSON format** as other converters
- **Compatible** with existing segmentation tools
- **Unified workflow** across different ECG data sources

## Usage:

1. **Update the file path** at the top of the script:

```python
input_file = "path/to/your/bard_ecg_file.txt"
```

2. **Run the converter**:

```bash
python bard_to_json.py
```

The script will:

- **Parse** Bard TXT file and extract the sampling rate
- **Convert** ADC values to millivolts using 16-bit parameters
- **Process** all 12 ECG leads
- **Validate** data quality and report any issues
- **Save** to a single JSON file with comprehensive metadata

# Compatibility:

This converter produces JSON files that are **fully compatible** with:

- The `json_segmenter.py` script for creating segments
- The same downstream analysis tools as other converters
- Consistent data structure across all ECG data sources

All three converters (Holter, China, Bard) now produce the same JSON structure, making it easy to build unified processing pipelines for different ECG data formats.

## JSON Segmenter (`json_segmenter.py`)

This script takes the single JSON file and segments it with configurable options:

### Features:

- **Flexible segmentation** - configurable segment duration
- **Optional scale factor** - apply additional scaling during segmentation
- **Command-line interface** with arguments
- **Progress tracking** and detailed output
- **Preserves metadata** for each segment

### Usage Examples:

```bash
# Basic usage (10-second segments, no additional scaling)
python json_segmenter.py ecg_complete_data.json

# Custom segment duration and output folder
python json_segmenter.py ecg_complete_data.json -o my_segments -d 30

# Apply additional scale factor during segmentation
python json_segmenter.py ecg_complete_data.json -s 0.5 -d 15

# Full example with all options
python json_segmenter.py ecg_complete_data.json -o processed_segments -d 20 -s 2.0
```

### Key Improvements:

- **Modular approach** - separate conversion and segmentation
- **Better metadata handling** - tracks scaling history and segment info
- **Flexible scaling** - can apply additional scale factors during segmentation
- **Command-line friendly** - easy to integrate into workflows
- **Progress feedback** - shows processing status

The segmenter maintains the same lead structure but adds more comprehensive metadata tracking and flexible configuration options.

## ECG Beat Extractor with Pan-Tompkins Algorithm

## Key Features:

### **Complete Pan-Tompkins Implementation**

- **Bandpass filtering** (5-15 Hz) to remove noise and baseline drift
- **Derivative filter** to emphasize QRS slope changes
- **Signal squaring** to amplify QRS complexes
- **Moving window integration** to smooth the signal
- **Adaptive thresholding** for robust peak detection

### **Robust Beat Extraction**

- **Multi-lead support** - extracts beats from all available leads
- **Configurable parameters** - beat window size, RR interval limits
- **Quality filtering** - removes beats with invalid RR intervals
- **Comprehensive metadata** - timing, heart rate, detection parameters

### **Output Structure**

- **Individual beat files** - each beat saved as separate JSON
- **Complete lead data** - all ECG leads preserved for each beat
- **Summary file** - overview of extraction results
- **Detection plots** - visual verification of algorithm performance

## Usage Examples:

### **Basic Usage:**

```bash
# Extract beats using default parameters (Lead II, 600ms window)
python beat_extractor.py your_ecg_file.json

# Specify output folder
python beat_extractor.py your_ecg_file.json -o my_beats_folder
```

### **Advanced Usage:**

```bash
# Use different lead for detection
python beat_extractor.py ecg_data.json -l V1 -o beats_v1

# Custom beat window and RR interval limits
python beat_extractor.py ecg_data.json -w 800 --min-rr 400 --max-rr 1500

# Generate detection plots for verification
python beat_extractor.py ecg_data.json --plot

# Full customization
python beat_extractor.py ecg_data.json -o custom_beats -l V2 -w 700 --min-rr 350 --max-rr 1800 --plot
```

## Output Structure:

### **Individual Beat Files:**

```json
{
  "metadata": {
    "beat_number": 0,
    "peak_index": 1250,
    "peak_time_seconds": 2.5,
    "rr_interval_ms": 750,
    "heart_rate_bpm": 80,
    "sampling_rate": 500,
    "detection_lead": "II"
  },
  "leads": {
    "I": [...],
    "II": [...],
    "V1": [...],
    // All available leads
  }
}
```

### **Summary File:**

```json
{
  "source_file": "ecg_data.json",
  "detection_parameters": {
    "detection_lead": "II",
    "beat_window_ms": 600,
    "sampling_rate": 500
  },
  "results": {
    "total_beats_detected": 45,
    "valid_beats_extracted": 42,
    "average_heart_rate_bpm": 75.5
  }
}
```

## Algorithm Details:

### **Pan-Tompkins Steps:**

1. **Bandpass Filter** - Removes baseline wander and high-frequency noise
2. **Derivative** - Emphasizes QRS slope information
3. **Squaring** - Amplifies QRS complexes, suppresses T-waves
4. **Integration** - Provides waveform feature information
5. **Adaptive Thresholding** - Dynamically adjusts detection thresholds

### **Quality Control:**

- **RR interval validation** - Filters physiologically impossible intervals
- **Beat window validation** - Ensures complete beat capture
- **Signal quality checks** - Warns about potential issues

### **Visualization:**

- **Multi-panel plots** showing each processing step
- **Peak detection overlay** on original signal
- **High-resolution output** for publication quality

## Compatibility:

The script works with **all your JSON formats**:

- ✅ **Holter binary** converted files
- ✅ **China DAT** converted files
- ✅ **Bard TXT** converted files
- ✅ **Segmented JSON** files from the segmenter

This gives you a complete offline pipeline for ECG beat extraction that's both robust and highly configurable!

# Wavelet ECG Beat Detection Script

## Wavelet Beat Extractor - Completion

Here's the completion of wavelet-enhanced ECG beat extraction script. The main function includes:

**Command Line Arguments:**

- Input file or directory for batch processing
- Output folder specification
- ECG lead selection
- Beat window size configuration
- RR interval constraints
- Wavelet denoising toggle
- Template matching parameters
- Plotting option
- Batch processing mode

**Key Features:**

- **Single file processing**: Process one ECG file at a time
- **Batch processing**: Process all JSON files in a directory with `--batch` flag
- **Flexible parameters**: All major processing parameters are configurable
- **Error handling**: Graceful handling of processing errors
- **Progress feedback**: Clear status messages during processing

**Usage Examples:**

```bash
# Single file with default settings
python wavelet_beat_extractor.py ecg_data.json

# Custom parameters with plotting
python wavelet_beat_extractor.py ecg_data.json --lead V1 --window 800 --plot

# Batch processing with custom correlation threshold
python wavelet_beat_extractor.py /path/to/ecg_files/ --batch --correlation-threshold 0.8

# Disable denoising and use more templates
python wavelet_beat_extractor.py ecg_data.json --no-denoising --templates 5
```

The script now provides a complete command-line interface for wavelet-enhanced ECG beat extraction system.
