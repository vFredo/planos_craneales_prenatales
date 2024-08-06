from roboflow.core.version import Dataset
from dataset import init_data_set
import os
import random
import sys

dataset = Dataset(name="planos-craneales-prenatales", version=2, model_format="yolov9", location="./augmented_dataset")

# Load your Roboflow dataset
option = input(" download database? [y/n]: ")
if option == 'y':
    dataset = init_data_set()

print("Name:", dataset.name)
print("Version:", dataset.version)
print("Model Format:", dataset.model_format)
print("Location:", dataset.location)

option = input(" continue balance? [y/n]: ")
if option == 'n':
    sys.exit(0)

print()

# Paths to downloaded images
train_dir = "Planos-craneales-prenatales-2/train"
val_dir = "Planos-craneales-prenatales-2/valid"
test_dir = "Planos-craneales-prenatales-2/test"

# Class names
class_names = ["trans_cerebellum", "trans_thalamic", "trans_ventricular"]

def balance_dataset(root_dir):
    print("=========", root_dir, "=========\n")

    images_dir = os.path.join(root_dir, "images")
    labels_dir = os.path.join(root_dir, "labels")

    class_images = { class_name: [] for class_name in class_names }

    for label_file in os.listdir(labels_dir):
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            class_id = int(f.readline().split()[0])
            class_name = class_names[class_id]

        img_file = label_file.replace(".txt", ".jpg")
        img_path = os.path.join(images_dir, img_file)

        if os.path.exists(img_path):
            class_images[class_name].append(label_file)
        else:
            print(f"Warning: Image {img_path} not found and will be skipped.")

    # Balance the dataset
    min_class_size = min([len(class_images[name]) for name in class_images])
    for class_name in class_images:
        curr_size = len(class_images[class_name])
        print(f" --> Class {class_name} has {curr_size} images before balancing")

        if curr_size > min_class_size:
            data_to_remove = random.sample(class_images[class_name], curr_size - min_class_size)

            for label_remove in data_to_remove:
                class_images[class_name].remove(label_remove)
                os.remove(os.path.join(labels_dir, label_remove))
                os.remove(os.path.join(images_dir, label_remove.replace(".txt", ".jpg")))

        curr_size = len(class_images[class_name])
        print(f" <--- Class {class_name} has {curr_size} images after balancing\n")


# Organize and balance the datasets
# balance_dataset(train_dir)
# balance_dataset(val_dir)
# balance_dataset(test_dir)
balance_dataset("augmented_dataset")

print("Dataset preparation and balancing completed.")
