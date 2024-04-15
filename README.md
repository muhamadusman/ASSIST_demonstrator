# ASSIST_demonstrator

docker build -t assist_container .                                     

docker run -v path-to-DICOM-Folder:/dicom -v path-to-Docker_Output:/finalOutput assist_container /dicom /finalOutput
