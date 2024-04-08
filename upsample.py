import os
import nibabel as nib
import numpy as np
import scipy.ndimage
import glob

def upsample_and_reorient_volume(nifti_file, original_shape, original_spacing, reference_volume_path, output_folder):
    """
    Upsample a NIfTI volume to its original shape and spacing, then reorient it to match
    the reference volume's orientation.
    """
    # Load the NIfTI file and get data and affine
    img = nib.load(nifti_file)
    data = img.get_fdata()
    affine = img.affine
    
    # Upsample volume
    resize_factor = np.array(original_shape) / np.array(data.shape)
    resampled_data = scipy.ndimage.zoom(data, resize_factor, order=0)
    if resampled_data.shape != original_shape:
        additional_resizing_factor = np.array(original_shape) / np.array(resampled_data.shape)
        resampled_data = scipy.ndimage.zoom(resampled_data, additional_resizing_factor, order=0)
    new_affine = np.copy(affine)
    np.fill_diagonal(new_affine, np.concatenate((original_spacing, [affine[3,3]])))
    upsampled_img = nib.Nifti1Image(resampled_data, new_affine)
    upsampled_img.header.set_zooms(original_spacing)
    
    # Load the reference volume for reorientation
    reference_img = nib.load(reference_volume_path)
    first_3d_volume_data = reference_img.dataobj[..., 0]
    first_3d_volume_img = nib.Nifti1Image(first_3d_volume_data, reference_img.affine, header=reference_img.header)
    
    # Reorient the upsampled volume
    reoriented_img = nib.Nifti1Image(upsampled_img.get_fdata(), first_3d_volume_img.affine, header=upsampled_img.header)

    # Save the reoriented upsampled volume
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, os.path.basename(nifti_file).replace('.nii', '_corrected.nii'))
    nib.save(reoriented_img, output_path)
    print(f"Processed volume saved to {output_path}")

def process_volumes(input_folder, output_folder, original_shape, original_spacing, reference_volume_path):
    """
    Process all NIfTI volumes in the input folder: upsample them and reorient to match a reference volume.
    """
    nifti_files = glob.glob(os.path.join(input_folder, '*.nii'))
    for file in nifti_files:
        upsample_and_reorient_volume(file, original_shape, original_spacing, reference_volume_path, output_folder)

# Parameters (Update these paths and values as needed)
input_folder = '/flush/muhak80/ASSIST_demo/C/Seg'
output_folder = '/flush/muhak80/ASSIST_demo/C/Seg_final'
original_shape = (392, 512, 512)  
original_spacing = (0.40000153, 0.4688, 0.4688)
reference_volume_path = '/flush/muhak80/ASSIST_demo/Data/DICOM_b/DICOM_Sag_3D_T1_a_20000101000101_5.nii'

# Process the volumes
process_volumes(input_folder, output_folder, original_shape, original_spacing, reference_volume_path)
