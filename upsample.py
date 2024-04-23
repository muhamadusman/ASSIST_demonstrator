import os
import numpy as np
import nibabel as nib
import scipy.ndimage
import argparse
import glob

def padAroundImageCenter(image, target_shape):
    """
    Pads with zeros or crops the image to match the target shape, centering the content.
    """
    current_shape = np.array(image.shape)
    padding = np.maximum(target_shape - current_shape, 0)
    pad_before = padding // 2
    pad_after = padding - pad_before
    padded_image = np.pad(image, [(pad_before[i], pad_after[i]) for i in range(3)], mode='constant', constant_values=0)
    
    # Crop if necessary
    crop_before = np.maximum(current_shape - target_shape, 0) // 2
    crop_after = crop_before + target_shape
    cropped_and_padded_image = padded_image[crop_before[0]:crop_after[0], crop_before[1]:crop_after[1], crop_before[2]:crop_after[2]]
    
    return cropped_and_padded_image

def find_nii_volume_with_name(directory, pattern):
    # Construct the search pattern
    search_pattern = os.path.join(directory, f"*{pattern}*.nii")

    # Use glob to find files matching the pattern
    matching_files = glob.glob(search_pattern)

    # Check if we found any matching files
    if matching_files:
        # For simplicity, return the first matching file's full path
        return matching_files[0]
    else:
        return None
    
def pad_and_resample_to_target(source_nifti_file, target_nifti_file, output_path):
    # Load the source and target volumes
    matching_volume = find_nii_volume_with_name(target_nifti_file, "Sag_3D_T1_GD")
    print ("Reference Volume : ",matching_volume)
    reference_volume_pa = os.path.join(target_nifti_file, matching_volume ) #"dicom_Sag_3D_T1_GD_a_20000101000101_7.nii")
    
    source_img = nib.load(source_nifti_file)
    target_img = nib.load(reference_volume_pa)
    
    # Extract data arrays and affines
    source_data = source_img.get_fdata()
    target_affine = target_img.affine
    target_header = target_img.header
    
    # Calculate resize factors based on voxel size differences
    source_spacing = np.array(source_img.header.get_zooms()[:3])
    target_spacing = np.array(target_img.header.get_zooms()[:3])
    resize_factor = source_spacing / target_spacing
    new_shape = target_img.shape
    resampled_data = scipy.ndimage.zoom(source_data, resize_factor, order=0)
    
    # Adjust the resampled_data shape to match the target exactly
    resampled_data = padAroundImageCenter(resampled_data, new_shape)

    # Create a new NIfTI image using the resampled data and the target's affine
    new_img = nib.Nifti1Image(resampled_data, target_affine, header=target_header)

    # Save the resampled and aligned volume
    nib.save(new_img, output_path)
    print(f"Resampling and alignment complete. The volume has been saved to {output_path}.")

def main():
    parser = argparse.ArgumentParser(description="Resample and align a source NIFTI volume to match a target volume.")
    parser.add_argument("source_nifti_file", help="Path to the source NIfTI volume.")
    parser.add_argument("target_nifti_file", help="Path to the target NIfTI volume for reference.")
    parser.add_argument("output_path", help="Path to save the resampled and aligned output volume.")

    args = parser.parse_args()

    pad_and_resample_to_target(args.source_nifti_file, args.target_nifti_file, args.output_path)

if __name__ == "__main__":
    main()
