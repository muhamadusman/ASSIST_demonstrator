import argparse
import os
from rt_utils import RTStructBuilder
import nibabel as nib
import numpy as np
import os
from datetime import datetime

# Example mapping of label numbers to custom names
# Update this dictionary based on your specific labels and desired ROI names
label_to_name = {
    1: "CTV",
    2: "GTV",
    3: "BRAINSTEM",
    # Add more mappings as needed
}

def generate_boolean_masks(mask_array):
    """
    Generate separate boolean masks for each unique ROI in the mask array.
    """
    masks = {}
    for label in np.unique(mask_array):
        if label == 0:  # Assuming 0 is the background
            continue
        masks[label] = (mask_array == label)
    return masks

def main(input_path, output_path):
    # Load the mask file
    mask_nifti = nib.load(input_path)
    mask_array = mask_nifti.get_fdata()

    # Move the first dimension to the end to get (512, 512, 392)
    rearranged_mask_array = mask_array.transpose(1, 2, 0)

    # Flip vertically the first two dimensions
    flipped_mask_array = np.flipud(rearranged_mask_array)

    # Rotate the first two dimensions 90 degrees anticlockwise
    rotated_mask_array = np.rot90(flipped_mask_array, 1, axes=(0, 1))

    # Create a new RT Struct
    rtstruct = RTStructBuilder.create_new(dicom_series_path="/tmp/singlemri")

    # Generate boolean masks for each ROI
    boolean_masks = generate_boolean_masks(rotated_mask_array)
        
    # Add each boolean mask as a separate ROI with custom names
    for label, boolean_mask in boolean_masks.items():
        # Use the label to get a custom name from the mapping, defaulting to "ROI_{label}" if not found
        roi_name = label_to_name.get(label, f"ROI_{label}")
        rtstruct.add_roi(mask=boolean_mask, name=roi_name)

    # Get the current date and time
    now = datetime.now()

    # Format the date and time in a preferred format, for example: 'YYYY-MM-DD_HH-MM-SS'
    date_time_format = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create the filename including the client name and the date and time
    filename = f"RTSTRUCT_{date_time_format}.dcm"
    
    # Save the new RT Struct
    savePth = os.path.join(output_path, filename)
    rtstruct.save(savePth)
    print("RT Struct saved successfully with multiple ROIs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RT Struct with multiple ROIs.")
    parser.add_argument("input_path", help="Path to the input mask file.")
    parser.add_argument("output_path", help="Path to save the output RT Struct file.")
    args = parser.parse_args()

    main(args.input_path, args.output_path)
