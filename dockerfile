# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install any necessary dependencies
RUN apt-get update && apt-get install -y \
    dcm2niix \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies required by the scripts
RUN pip install numpy nibabel scipy

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the scripts and orchestration script into the container
COPY convert.py getnifty.py process_volumes.sh ./

# Make the orchestration script executable
RUN chmod +x process_volumes.sh

# Define environment variables if needed
# ENV ...

# The command to run the orchestration script
CMD ["./process_volumes.sh"]

COPY process_volumes.sh .
RUN chmod +x process_volumes.sh
CMD ["./process_volumes.sh"]
