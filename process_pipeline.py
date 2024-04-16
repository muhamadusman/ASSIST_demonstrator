import os
import subprocess
import sys

import logging
import logging.handlers
import sys

import logging
import os
from logging.handlers import TimedRotatingFileHandler
import sys


def run_convert_dicom_to_nifti(dicom_folder, tmp_nifti_folder):
    subprocess.run(["python", "convert.py", dicom_folder, tmp_nifti_folder], check=True)

def run_get_nifty(tmp_nifti_folder, tmp_sorted_Nifty, tmp_processed_nifti_folder, itmp_final):
    subprocess.run(["python", "getnifty.py",
                    "--source_dir", tmp_nifti_folder,
                    "--temp_dir", tmp_sorted_Nifty,
                    "--resampled_dir", tmp_processed_nifti_folder,
                    "--final_dir", itmp_final], check=True)

def run_inference(processed_nifti_folder, final_output_folder):
    # Adjust this command based on how you expect to run inference.py and where its inputs/outputs are configured
    subprocess.run(["python", "inference.py", 
                    "--input_volume_path", os.path.join(processed_nifti_folder, "final_nifti_volume.nii"), 
                    "--output_folder", final_output_folder], check=True)
    
def run_singlemri(dicom_folder, singlemri):
    subprocess.run(["python", "singlemri.py",
                    dicom_folder,  # Positional argument for input_folder
                    singlemri],    # Positional argument for output_folder
                   check=True)

def run_Convert_RTSTRUCT(Segmenation_Mask, final_output_folder_path):
    mask_volume = os.path.join(Segmenation_Mask, "tmp_segmentation_corrected.nii")
    print("OutPut Directory : ", final_output_folder_path)
    subprocess.run(["python", "RTSTRUCT.py", 
                    mask_volume, 
                    final_output_folder_path], check=True)
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_pipeline.py <dicom_folder_path> <final_output_folder_path>")
        sys.exit(1)
    
    dicom_folder_path = sys.argv[1]
    final_output_folder_path = sys.argv[2]
    # print ("InputPath : ", dicom_folder_path)
    # print ("OutputPath : ", final_output_folder_path)
    
    
    tmp_nifti_folder = "/data/"
    tmp_sorted_Nifty = "/tmp/tmp_sorted_Nifty/"
    tmp_resampled_nifty_folder = "/tmp/resampled_nifty/"
    itmp_final = "/tmp/final_nifty/"
    tmp_mask = "/output/"
    tmp_singleMRI = "/tmp/singlemri"

    # # Ensure intermediate directories exist
    os.makedirs(tmp_resampled_nifty_folder, exist_ok=True)
    os.makedirs(itmp_final, exist_ok=True)
    os.makedirs(tmp_nifti_folder, exist_ok=True)
    os.makedirs(tmp_sorted_Nifty, exist_ok=True)
    os.makedirs(final_output_folder_path, exist_ok=True)
    os.makedirs(tmp_singleMRI, exist_ok=True)
    
    # Step 1: Convert DICOM to NIfTI and save a singlemri sequence
    run_convert_dicom_to_nifti(dicom_folder_path, tmp_nifti_folder)
    run_singlemri(dicom_folder_path, tmp_singleMRI)
    
    
    # testimage = "/app/test_image/test.nii"
    # Step 2: Get useful NIfTI volume
    print (tmp_nifti_folder)
    print (tmp_sorted_Nifty)
    run_get_nifty(tmp_nifti_folder, tmp_sorted_Nifty, tmp_resampled_nifty_folder, itmp_final)
    
    # Step 3: Perform inference
    run_inference(itmp_final, tmp_mask) # tmp_mask
    
    # #Step 4 : Generate RTSTRUCT Mask
    run_Convert_RTSTRUCT(tmp_mask,final_output_folder_path)
    
    
    
    # Define Complete Flag
    file_name = os.path.join(final_output_folder_path, "completed.txt")
    text_to_write = "Completed"
    with open(file_name, 'w') as file:
        file.write(text_to_write)
        
    print(f"'{text_to_write}' has been written to {file_name}")
    
    # Define the base directory for your logs




