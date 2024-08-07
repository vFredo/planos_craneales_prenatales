import os
import shutil
import random

def split_dataset(source_dir, dest_dir, split_ratios=(0.7, 0.2, 0.1)):
    # Create destination directories
    for split in ['train', 'test', 'valid']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(dest_dir, split, subdir), exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(os.path.join(source_dir, 'images')) if f.endswith('.jpg')]
    
    # Shuffle the list of image files
    random.shuffle(image_files)
    
    # Calculate split indices
    total = len(image_files)
    train_end = int(total * split_ratios[0])
    test_end = train_end + int(total * split_ratios[1])
    
    # Split and copy files
    for i, img_file in enumerate(image_files):
        label_file = img_file.replace(".jpg", ".txt")
        
        if i < train_end:
            split = 'train'
        elif i < test_end:
            split = 'test'
        else:
            split = 'valid'
        
        # Copy image
        shutil.copy2(os.path.join(source_dir, 'images', img_file), 
                     os.path.join(dest_dir, split, 'images', img_file))
        
        # Copy label
        shutil.copy2(os.path.join(source_dir, 'labels', label_file), 
                     os.path.join(dest_dir, split, 'labels', label_file))
    
    print(f"Dataset split complete. Total files: {total}")
    print(f"Train: {train_end}, Test: {test_end - train_end}, Valid: {total - test_end}")

split_dataset('augmented_dataset', 'yolo_dataset')