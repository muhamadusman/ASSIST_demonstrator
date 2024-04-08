#!/bin/bash

# Exit on errors
set -e

# Directory setup
DICOM_DIR="/data/dicom"
NIFTI_TEMP_DIR="/data/nifti_temp"
PROCESSING_TEMP_DIR="/data/processing_temp"  # Only needed if separate processing steps require it
RESAMPLED_DIR="/data/resampled"  # Can be same as PROCESSING_TEMP_DIR if you prefer
FINAL_DIR="/data/final"

# Ensure all directories exist
mkdir -p "$NIFTI_TEMP_DIR" "$PROCESSING_TEMP_DIR" "$RESAMPLED_DIR" "$FINAL_DIR"

echo "Converting DICOM to NIfTI format..."
# Convert DICOM to NIfTI
python convert.py "$DICOM_DIR" "$NIFTI_TEMP_DIR"

echo "Processing NIfTI volumes..."
# Process the NIfTI volumes to generate the final volume
python getnifty.py --source_dir "$NIFTI_TEMP_DIR" --temp_dir "$PROCESSING_TEMP_DIR" --resampled_dir "$RESAMPLED_DIR" --final_dir "$FINAL_DIR"

echo "Processing complete. Final volume is in $FINAL_DIR."
