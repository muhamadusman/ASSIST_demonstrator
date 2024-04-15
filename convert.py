import os
import subprocess
import sys

def convert_dicom_to_nifti(dicom_folder, output_folder):
    """
    Convert DICOM images in a folder to NIfTI format using dcm2niix.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Run dcm2niix
    subprocess.run(["dcm2niix", "-o", output_folder, dicom_folder], check=True)
    print ("Nifty Volumes saved to : ", output_folder)
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py <dicom_folder> <output_folder>")
        sys.exit(1)

    dicom_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    
    convert_dicom_to_nifti(dicom_folder, output_folder)
