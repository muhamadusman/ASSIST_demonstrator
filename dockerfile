FROM python:3.8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    dcm2niix \
    libc6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    dcmtk \
 && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    nibabel \
    monai \
    scipy \
    pydicom \
    SimpleITK \
    matplotlib \
    pydicom \
    rt_utils \
    scikit-image 

# Copy the scripts and any other necessary files into the image
COPY . /app
WORKDIR /app

# Make sure the main script is executable
RUN chmod +x process_pipeline.py

# Set the ENTRYPOINT to run your master script
ENTRYPOINT ["python", "process_pipeline.py"]

