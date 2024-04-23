import os
import torch
import numpy as np
import nibabel as nib
import scipy.ndimage
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, EnsureTyped
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
import collections
import argparse
import glob
import shutil
from scipy.ndimage import zoom

def get_transform():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ])

def load_model_from_npz(model_path, device):
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=4,
        dropout_prob=0.2,
    ).to(device)

    # Load .npz file containing the weights
    npz_weights = np.load(model_path, allow_pickle=True)
    model_state_dict = model.state_dict()

    new_state_dict = collections.OrderedDict()
    for i, (key, value) in enumerate(model_state_dict.items()):
        layer_weights = torch.from_numpy(npz_weights[str(i)])
        new_state_dict[key] = layer_weights

    model.load_state_dict(new_state_dict)
    return model

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

def upsample_and_reorient_volume(nifti_file, original_shape, original_spacing, reference_volume_path, output_folder):
    img = nib.load(nifti_file)
    data = img.get_fdata()
    affine = img.affine

    resize_factor = np.array(original_shape) / np.array(data.shape)
    resampled_data = scipy.ndimage.zoom(data, resize_factor, order=0)

    if resampled_data.shape != original_shape:
        additional_resizing_factor = np.array(original_shape) / np.array(resampled_data.shape)
        resampled_data = scipy.ndimage.zoom(resampled_data, additional_resizing_factor, order=0)

    new_affine = np.copy(affine)
    np.fill_diagonal(new_affine, np.concatenate((original_spacing, [affine[3,3]])))

    upsampled_img = nib.Nifti1Image(resampled_data, new_affine)
    upsampled_img.header.set_zooms(original_spacing)

    reference_img = nib.load(reference_volume_path)
    first_3d_volume_data = reference_img.dataobj[..., 0]
    first_3d_volume_img = nib.Nifti1Image(first_3d_volume_data, reference_img.affine, header=reference_img.header)

    reoriented_img = nib.Nifti1Image(upsampled_img.get_fdata(), first_3d_volume_img.affine, header=upsampled_img.header)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, os.path.basename(nifti_file).replace('.nii', '_Upsampled.nii'))
    target_shape =  (392, 512, 512)
    nib.save(reoriented_img, output_path)    
    # resample_volume (output_path, output_folder, target_shape)
    print(f"Processed volume saved to {output_path}")

def upsample_and_reorient_volume_0(nifti_file, original_shape, original_spacing, reference_volume_path, output_folder):
    img = nib.load(nifti_file)
    data = img.get_fdata()
    affine = img.affine

    resize_factor = np.array(original_shape) / np.array(data.shape)
    resampled_data = scipy.ndimage.zoom(data, resize_factor, order=0)

    if resampled_data.shape != original_shape:
        additional_resizing_factor = np.array(original_shape) / np.array(resampled_data.shape)
        resampled_data = scipy.ndimage.zoom(resampled_data, additional_resizing_factor, order=0)

    new_affine = np.copy(affine)
    np.fill_diagonal(new_affine, np.concatenate((original_spacing, [affine[3,3]])))

    upsampled_img = nib.Nifti1Image(resampled_data, new_affine)
    upsampled_img.header.set_zooms(original_spacing)

    reference_img = nib.load(reference_volume_path)
    first_3d_volume_data = reference_img.dataobj[..., 0]
    first_3d_volume_img = nib.Nifti1Image(first_3d_volume_data, reference_img.affine, header=reference_img.header)

    reoriented_img = nib.Nifti1Image(upsampled_img.get_fdata(), first_3d_volume_img.affine, header=upsampled_img.header)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, os.path.basename(nifti_file).replace('.nii', '_downsampled.nii'))
    nib.save(reoriented_img, output_path)
    print(f"Processed volume saved to {nifti_file}")
    
    # matching_volume = find_nii_volume_with_name(ref_nifty_path, "Sag_3D_T1_a")
    # print ("Reference Volume : ",matching_volume)
    # reference_volume_pa = os.path.join(ref_nifty_path, matching_volume ) #"dicom_Sag_3D_T1_GD_a_20000101000101_7.nii")
    # pad_and_resample_to_target(nifti_file, reference_volume_pa, output_folder)
    

def pad_and_resample_to_target(source_nifti_file, target_nifti_file, output_path):
    # Load the source and target volumes
    source_img = nib.load(source_nifti_file)
    target_img = nib.load(target_nifti_file)
    
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
    print ("Usamplede Segmentation Mask saved at ",output_path)
    tmp_output_path = os.path.join(output_folder, "tmp_segmentation_Upsampled.nii")
    nib.save(new_img, tmp_output_path)
    
    print(f"Resampling and alignment complete. The volume has been saved to {output_path}.")
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

def predict_and_process_volume(input_volume_path, model, device, transform, original_shape, original_spacing, reference_volume_path, output_folder):
    data = {"image": input_volume_path}
    data = transform(data)
    img = data["image"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = sliding_window_inference(img, (256, 256, 120), 4, model, overlap=0.5)
    segmentation_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # ref_img_ = nib.load(input_volume_path)
    # header_ = ref_img_.header
    voxel_spacing_ = (1,1,2) #header_.get_zooms()
    volume_size_ = (256,256,140)#ref_img_.shape
    
    tmp_output_path = os.path.join(output_folder, "tmp_segmentation.nii")
    print ("Final Segmentation mask saved : ", tmp_output_path)
    nib.save(nib.Nifti1Image(segmentation_mask.astype(np.uint8), np.eye(4)), tmp_output_path)
    
    
    upsample_and_reorient_volume_0(tmp_output_path, volume_size_, voxel_spacing_, input_volume_path, output_folder)
    # pad_and_resample_to_target(tmp_output_path, reference_volume_path, output_folder)
    # upsample_and_reorient_volume(tmp_output_path, original_shape, original_spacing, reference_volume_path, output_folder)
    # os.remove(tmp_output_path)
  

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a NIfTI volume.")
    parser.add_argument("--input_volume_path", required=True, help="Path to the input NIfTI volume.")
    parser.add_argument("--output_folder", required=True, help="Directory to save the output.")
    
    args = parser.parse_args()
    
    base_path = "/appdata/" 
    ref_nifty_path = "/data/"
    input_volume_path = args.input_volume_path
    output_folder = args.output_folder
    
    print ("Input Volume  : ", input_volume_path)
    print (" Output volume Path : ", output_folder)
    # input_volume = os.path.join(input_volume_path, "final_nifti_volume.nii")
    model_path = os.path.join(base_path, "Model_Weights.npz")
    
    matching_volume = find_nii_volume_with_name(ref_nifty_path, "Sag_3D_T1_a")
    print ("Reference Volume : ",matching_volume)
    reference_volume_path = os.path.join(ref_nifty_path, matching_volume ) #"dicom_Sag_3D_T1_GD_a_20000101000101_7.nii")
    
    #Copy the niftyVolume to output directory
    
    copy_path = os.path.join(output_folder , matching_volume )
    print("Path to Copy to",copy_path)
    shutil.copy2(reference_volume_path, output_folder)
    
    #getting the size and voxel_spacing
    ref_img = nib.load(reference_volume_path)
    header = ref_img.header
    voxel_spacing = header.get_zooms()
    volume_size = ref_img.shape
        
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # volume_size = (512, 512, 392)  # Adjust as per your requirements
    # voxel_spacing = (0.4688, 0.4688,40000153)  # Adjust as per your requirements
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform()
    model = load_model_from_npz(model_path, device)
    predict_and_process_volume(input_volume_path, model, device, transform, volume_size, voxel_spacing, reference_volume_path, output_folder)
    
