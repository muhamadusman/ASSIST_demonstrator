#!/bin/bash

# Ensure the script exits on any error
set -e

# Check if exactly two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path1> <path2>"
    exit 1
fi

# Call the Python script with the provided paths
python /appdata/process_pipeline.py "$1" "$2"
