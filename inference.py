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
    output_path = os.path.join(output_folder, os.path.basename(nifti_file).replace('.nii', '_corrected.nii'))
    nib.save(reoriented_img, output_path)
    print(f"Processed volume saved to {output_path}")

def predict_and_process_volume(input_volume_path, model, device, transform, original_shape, original_spacing, reference_volume_path, output_folder):
    data = {"image": input_volume_path}
    data = transform(data)
    img = data["image"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = sliding_window_inference(img, (256, 256, 120), 4, model, overlap=0.5)
    segmentation_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    tmp_output_path = os.path.join(output_folder, "tmp_segmentation.nii")
    print ("Final Segmentation mask saved : ", tmp_output_path)
    nib.save(nib.Nifti1Image(segmentation_mask.astype(np.uint8), np.eye(4)), tmp_output_path)

    upsample_and_reorient_volume(tmp_output_path, original_shape, original_spacing, reference_volume_path, output_folder)
    os.remove(tmp_output_path)
    
    print ("******")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a NIfTI volume.")
    parser.add_argument("--input_volume_path", required=True, help="Path to the input NIfTI volume.")
    parser.add_argument("--output_folder", required=True, help="Directory to save the output.")
    
    args = parser.parse_args()
    
    base_path = "/app/" 
    ref_nifty_path = "/tmp/nifti/"
    input_volume_path = args.input_volume_path
    output_folder = args.output_folder
    
    # input_volume = os.path.join(input_volume_path, "final_nifti_volume.nii")
    model_path = os.path.join(base_path, "Model_Weights.npz")
    reference_volume_path = os.path.join(ref_nifty_path, "dicom_Sag_3D_T1_GD_a_20000101000101_7.nii")

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    original_shape = (392, 512, 512)  # Adjust as per your requirements
    original_spacing = (0.40000153, 0.4688, 0.4688)  # Adjust as per your requirements
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform()
    model = load_model_from_npz(model_path, device)
    predict_and_process_volume(input_volume_path, model, device, transform, original_shape, original_spacing, reference_volume_path, output_folder)
    
    
    
    #final_nifti_volume.nii