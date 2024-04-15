
import os
import shutil
import argparse
import pydicom

def get_file_type(dicom_file):
    # Use pydicom to read the DICOM file
    ds = pydicom.dcmread(dicom_file)
    
    # Example: Retrieve Modality and Series Description as file type
    file_type = ds.get("Modality", "") + " " + ds.get("SeriesDescription", "")
    return file_type

def main(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop over DICOM files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".dcm"):
            dicom_file = os.path.join(input_folder, filename)
            try:
                file_type = get_file_type(dicom_file)
                # print(file_type)
                if "Sag 3D T1 GD" in file_type:
                    shutil.copy(dicom_file, output_folder)
                    # print(f"Copied {dicom_file} to {output_folder}")
                elif "Sagadsdsad" in file_type:
                    pass  # You can add additional logic here for other file types
            except Exception as e:
                print(f"Error processing {dicom_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort DICOM files by sequence type.")
    parser.add_argument("input_folder", help="Path to the input DICOM folder.")
    parser.add_argument("output_folder", help="Path to the output folder for the sorted DICOM files.")
    args = parser.parse_args()

    main(args.input_folder, args.output_folder)
    
