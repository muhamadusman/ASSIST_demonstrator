import os
import shutil
import numpy as np
import nibabel as nib
import scipy.ndimage
import glob
import sys
import argparse

def padAroundImageCenter(imageArray, paddedSize):
    origShape = imageArray.shape
    diff = (np.array(paddedSize) - np.array(origShape)) / 2
    extraPadding = [(int(np.floor(d)), int(np.ceil(d))) for d in diff]
    paddedImageArray = np.pad(imageArray, extraPadding, mode='constant', constant_values=0)
    return paddedImageArray

def flip_volume_if_needed(nifti_file, data):
    specific_text = "Ax_T2_GD"
    filename = os.path.basename(nifti_file)
    
    if specific_text in filename:
        print(f"Flipping {filename} vertically before resampling.")
        data = np.flip(data, axis=0)  # Flipping operation
        
    return data


def resample_and_reshape_volume(nifti_file, new_shape, new_spacing, output_folder):
    # specific_file_to_flip = "dicom_Ax_T2_GD_a_20000101000101_6.nii"  # Ensure name matches exactly
    
    img = nib.load(nifti_file)
    data = img.get_fdata()
    affine = img.affine

    # Debugging print statement
    print(f"Processing file: {os.path.basename(nifti_file)}")

    # if os.path.basename(nifti_file) == specific_file_to_flip:
    #     print(f"Flipping {specific_file_to_flip} vertically before resampling.")
    #     data = np.flip(data, axis=0)  # Flipping operation
    specific_text = "Ax_T2_GD"
    filename = os.path.basename(nifti_file)
    
    if specific_text in filename:
        print(f"Flipping {filename} vertically before resampling.")
        data = np.flip(data, axis=0)  # Flipping operation
        
    
    current_spacing = np.array(img.header.get_zooms()[:3])
    resize_factor = current_spacing / new_spacing
    new_real_shape = data.shape * resize_factor
    resampled_data = scipy.ndimage.zoom(data, resize_factor, order=2)

    resampled_data = padAroundImageCenter(resampled_data, new_shape)

    new_affine = np.copy(affine)
    np.fill_diagonal(new_affine, list(new_spacing) + [1])
    new_img = nib.Nifti1Image(resampled_data, new_affine)
    output_path = os.path.join(output_folder, os.path.basename(nifti_file))
    nib.save(new_img, output_path)

def reorient_to_standard(input_volume_path):
    # Load the original volume data
    volume_nii = nib.load(input_volume_path)

    # Reorient the volume to RAS+ orientation (Right, Anterior, Superior)
    volume_nii = nib.as_closest_canonical(volume_nii)

    return volume_nii

def process_volumes_in_folder(input_folder, output_folder, new_shape, new_spacing):
    nifti_files = [f for f in glob.glob(os.path.join(input_folder, '*.nii'))]
    if not nifti_files:
        print(f"No NIfTI files found in {input_folder}. Exiting.")
        sys.exit(1)

    for nifti_file in nifti_files:
        resample_and_reshape_volume(nifti_file, new_shape, new_spacing, output_folder)

# def process_volumes_in_folder(input_folder, output_folder, new_shape, new_spacing):
#     nifti_files = [f for f in glob.glob(os.path.join(input_folder, '*.nii'))]
#     if not nifti_files:
#         print(f"No NIfTI files found in {input_folder}. Exiting.")
#         sys.exit(1)

#     # Substrings to match in the filenames
#     substrings = ["Sag_3D_T1_GD", "Sag_3D_T2_flair", "Sag_3D_T1"]

#     for nifti_file in nifti_files:
#         # Check if the file name contains any of the specified substrings
#         if any(substring in os.path.basename(nifti_file) for substring in substrings):
#             # Reorient the volume to standard anatomical orientation before resampling and reshaping
#             reoriented_nii = reorient_to_standard(nifti_file)

#             # Proceed with resampling and reshaping
#             resample_and_reshape_volume(reoriented_nii, new_shape, new_spacing, output_folder)




import os
import glob
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def resample_volume(nifti_file, output_folder, target_shape):
    img = nib.load(nifti_file)
    data = img.get_fdata()
    
    # Calculate new zooms to get to the target shape
    current_shape = np.array(data.shape)
    target_shape = np.array(target_shape)
    new_zooms = target_shape / current_shape
    
    # Use scipy's zoom function to resample the volume data
    resampled_data = zoom(data, new_zooms, order=3)  # Cubic interpolation
    
    # Prepare the new NIfTI image
    new_affine = np.copy(img.affine)
    # Adjust the affine matrix to account for the new zooms
    scaling_affine = np.diag(new_zooms.tolist() + [1])
    new_affine = np.dot(new_affine, np.linalg.inv(scaling_affine))
    
    new_img = nib.Nifti1Image(resampled_data, new_affine)
    
    # Save the new volume
    output_path = os.path.join(output_folder, os.path.basename(nifti_file))
    nib.save(new_img, output_path)

def process_volumes_in_folder_0(input_folder, output_folder, target_shape, text_fragments):
    nifti_files = glob.glob(os.path.join(input_folder, '*.nii'))
    
    if not nifti_files:
        print(f"No NIfTI files found in {input_folder}. Exiting.")
        return
    
    for nifti_file in nifti_files:
        # Check if the file name contains any of the specified substrings
        if any(fragment in os.path.basename(nifti_file) for fragment in text_fragments):
            resample_volume(nifti_file, output_folder, target_shape)
            print(f"Processed {os.path.basename(nifti_file)}")
        else:
            print(f"Skipped {os.path.basename(nifti_file)} due to name mismatch")




def concatenate_nifti_volumes(input_folder, output_path):
    nifti_files = reversed(sorted(glob.glob(os.path.join(input_folder, '*.nii'))))
    if not nifti_files:
        print(f"No NIfTI files found in {input_folder} for concatenation. Exiting.")
        sys.exit(1)

    # volume_to_flip = "Dicom_Ax_T2_GD_a_20000101000101_6.nii"
    concatenated_data = []

    for nifti_file in nifti_files:
        img = nib.load(nifti_file)
        img_data = img.get_fdata()
        # if os.path.basename(nifti_file) == volume_to_flip:
            # img_data = np.flip(img_data, axis=0)
        concatenated_data.append(img_data)

    concatenated_data = np.stack(concatenated_data, axis=-1)
    concatenated_img = nib.Nifti1Image(concatenated_data, img.affine)
    nib.save(concatenated_img, output_path)
    print(f"Saved concatenated volume to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process NIfTI volumes.')
    parser.add_argument('--source_dir', required=True, help='Directory containing source NIfTI files.')
    parser.add_argument('--temp_dir', required=True, help='Temporary directory for intermediate processing.')
    parser.add_argument('--resampled_dir', required=True, help='Directory for resampled NIfTI volumes.')
    parser.add_argument('--final_dir', required=True, help='Directory to save the final NIfTI volume.')
    args = parser.parse_args()

    source_dir = args.source_dir
    temp_dir = args.temp_dir
    resampled_dir = args.resampled_dir
    final_dir = args.final_dir

    text_fragments = ["dicom_Sag_3D_T1_GD", "dicom_Sag_3D_T2_flair", "dicom_Sag_3D_T1", "dicom_Ax_T2_GD"]
    new_shape = (256, 256, 140)
    new_spacing = (1.0, 1.0, 2.0)

    # os.makedirs(temp_dir, exist_ok=True)
    # os.makedirs(resampled_dir, exist_ok=True)
    # os.makedirs(final_dir, exist_ok=True)


    # Step 1: Copy relevant .nii files to temp directory
    for filename in os.listdir(source_dir):
        if any(text in filename for text in text_fragments) and filename.endswith('.nii'):
            shutil.copy2(os.path.join(source_dir, filename), os.path.join(temp_dir, filename))
        else:
            print(f"Skipping file not matching criteria: {filename}")


    target_shape = (512, 512, 392)  # The desired new shape
    text_fragments = ["Sag_3D_T1_GD", "Sag_3D_T2_flair", "Sag_3D_T1"]

    process_volumes_in_folder_0(temp_dir, temp_dir, target_shape, text_fragments)
    
    # Step 2: Resample and reshape
    process_volumes_in_folder(temp_dir, resampled_dir, new_shape, new_spacing)
    
    # Step 3: Concatenate
    final_output_path = os.path.join(final_dir, 'final_nifti_volume.nii')
    concatenate_nifti_volumes(resampled_dir, final_output_path)
