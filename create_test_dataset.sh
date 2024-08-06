#!/bin/bash

# Check if a directory is provided
if [ $# -eq 0 ]; then
    echo "Please provide a directory path"
    exit 1
fi

# Change to the specified directory
cd "$1" || exit

# Check if 'images' and 'labels' subdirectories exist
if [ ! -d "images" ] || [ ! -d "labels" ]; then
    echo "Both 'images' and 'labels' subdirectories must exist"
    exit 1
fi

# Count total number of files in 'images' directory
total_files=$(find ./images -maxdepth 1 -type f | wc -l)

# Calculate number of files to keep (1% of total)
files_to_keep=$((total_files * 1 / 100))

# Shuffle files in 'images' and keep 20%
find ./images -maxdepth 1 -type f | shuf -n "$files_to_keep" > keep_files.txt

mkdir -p images_keep labels_keep

# Remove files not in the keep_files.txt list from both 'images' and 'labels'
while IFS= read -r file
do
    basename=$(basename "$file")
    label_file="./labels/${basename%.*}.txt"

    # Keep the files in both directories
    mv "$file" "./images_keep/"
    mv "$label_file" "./labels_keep/"
done < keep_files.txt

# Remove the remaining files
rm -r ./images/*
rm -r ./labels/*

# Move kept files back
mv ./images_keep/* ./images/
mv ./labels_keep/* ./labels/

# Clean up
rm keep_files.txt
rmdir images_keep labels_keep

echo "Approximately $files_to_keep file pairs have been removed."
