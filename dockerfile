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
COPY . /appdata
WORKDIR /appdata

RUN mkdir -p /logs /data /output

# Make the Bash script executable
RUN chmod +x ./entrypoint.sh

# Run the Bash script when the container launches
ENTRYPOINT ["./entrypoint.sh"]

