import os
import shutil

train_dir = "./yolo_dataset/train"
val_dir = "./yolo_dataset/valid"
test_dir = "./yolo_dataset/test"

# Move images to class-specific directories based on labels
def organize_dataset(root_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    images_dir = os.path.join(root_dir, "images")
    labels_dir = os.path.join(root_dir, "labels")

    for label_file in os.listdir(labels_dir):
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            class_id = int(f.readline().split()[0])
            class_name = ["trans_cerebellum", "trans_thalamic", "trans_ventricular"][class_id]

            img_file = label_file.replace(".txt", ".jpg")  # Assuming images are in .jpg format
            img_path = os.path.join(images_dir, img_file)
            class_dir = os.path.join(dest_dir, class_name)

            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            if os.path.exists(img_path):
                shutil.copy2(img_path, os.path.join(class_dir, img_file))
            else:
                print(f"Warning: Image {img_path} not found and will be skipped.")

# Organize the datasets
organize_dataset(train_dir, "resnet_dataset/train")
organize_dataset(val_dir, "resnet_dataset/valid")
organize_dataset(test_dir, "resnet_dataset/test")