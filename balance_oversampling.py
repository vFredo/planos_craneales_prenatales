import os
import cv2
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# Define augmentation pipeline
seq = iaa.Sequential([
    # Flip: Horizontal, Vertical (already have 50% chance)
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    # 0 to n of the pixels in an image with salt and pepper noise
    iaa.SaltAndPepper((0, 0.0022)),
    # 90° Rotate: Clockwise, Counter-Clockwise, Upside Down
    iaa.Sometimes(0.4, iaa.Rotate([0, 90, 180, 270])),
    # Crop: 0% Minimum Zoom, 30% Maximum Zoom
    iaa.Sometimes(0.3, iaa.Crop(percent=(0, 0.3))),
    # Brightness: Between -15% and +15%
    iaa.Sometimes(0.4, iaa.Multiply((0.85, 1.15))),
    # Blur: Up to 2.5px
    iaa.Sometimes(0.4, iaa.GaussianBlur(sigma=(0, 2.5))),
    # Traslation: 20% of the width and 20% of the height from the middle point
    iaa.Sometimes(0.4, iaa.Affine(
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        mode='constant',
        cval=0
    )),
])

def yolo_to_bbox(yolo_annotation, img_width, img_height):
    class_id, x_center, y_center, width, height = map(float, yolo_annotation.split())
    x_min = int((x_center - width/2) * img_width)
    y_min = int((y_center - height/2) * img_height)
    x_max = int((x_center + width/2) * img_width)
    y_max = int((y_center + height/2) * img_height)
    return BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max, label=int(class_id))

def bbox_to_yolo(bbox, img_width, img_height):
    x_center = (bbox.x1 + bbox.x2) / (2 * img_width)
    y_center = (bbox.y1 + bbox.y2) / (2 * img_height)
    width = (bbox.x2 - bbox.x1) / img_width
    height = (bbox.y2 - bbox.y1) / img_height
    return f"{bbox.label} {x_center} {y_center} {width} {height}"


def min_class_elements(images_dir, labels_dir):

    class_labels = ["trans_cerebellum", "trans_thalamic", "trans_ventricular"]
    class_images = { class_name: [] for class_name in class_labels }

    for label_file in os.listdir(labels_dir):
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            class_id = int(f.readline().split()[0])
            class_name = class_labels[class_id]

        img_file = label_file.replace(".txt", ".jpg")
        img_path = os.path.join(images_dir, img_file)

        if os.path.exists(img_path):
            class_images[class_name].append(img_file)
        else:
            print(f"Warning: Image {img_path} not found and will be skipped.")

    for name in class_images:
        print(f"{name} : {len(class_images[name])}")

    min_size = min([len(class_images[name]) for name in class_images])
    return min_size, class_images

def make_augmentation(name, max_limit, num_augmentations, images_dir, labels_dir, output_dir, class_images):

    valid_count = 0
    size_class = len(class_images[name])
    limit_augmentations = 0 if max_limit - size_class < 0 else max_limit - size_class
    print("--> class:", name ,"limit:", limit_augmentations)

    for img_file in class_images[name]:
        if limit_augmentations == 0:
            break

        image = cv2.imread(os.path.join(images_dir, img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_file = img_file.replace(".jpg", ".txt")
        label_path = os.path.join(labels_dir, label_file)

        with open(label_path, 'r') as f:
            yolo_annotation = f.read().strip()

        # Convert YOLO annotation to bounding box format
        bbox = yolo_to_bbox(yolo_annotation, image.shape[1], image.shape[0])
        bbs = BoundingBoxesOnImage([bbox], shape=image.shape)

        i = 0
        tries = 0
        while i < num_augmentations and limit_augmentations > 0 and tries <= 15:
            aug_image, aug_bbs = seq(image=image, bounding_boxes=bbs)
            bbox = aug_bbs.bounding_boxes[0]

            # Check if the bounding box is valid in reference of the size of the image
            if (0 <= bbox.x1 < aug_image.shape[1] and 0 <= bbox.x2 < aug_image.shape[1]
                and 0 <= bbox.y1 < aug_image.shape[0] and 0 <= bbox.y2 < aug_image.shape[0]):
                # Generate filenames
                aug_img_file = f"{img_file[0:-4]}_aug{i+1}.jpg"
                aug_label_file = f"{label_file[0:-4]}_aug{i+1}.txt"

                # Save augmented image
                cv2.imwrite(os.path.join(output_dir, 'images', aug_img_file),
                            cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

                # Save label file
                with open(os.path.join(output_dir, 'labels', aug_label_file), 'w') as f:
                    yolo_ann = bbox_to_yolo(bbox, aug_image.shape[1], aug_image.shape[0])
                    f.write(yolo_ann + '\n')

                limit_augmentations -= 1
                valid_count += 1
                i += 1
            else:
                tries += 1

    return valid_count

# Balance the dataset
def balance_dataset(root_dir, output_dir, num_augmentations):
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    images_dir = os.path.join(root_dir, "images")
    labels_dir = os.path.join(root_dir, "labels")
    min_size, class_images = min_class_elements(images_dir, labels_dir)
    max_limit = (min_size * num_augmentations) + min_size
    print("max_limit:", max_limit)

    for name in class_images:
        valid_count = make_augmentation(name, max_limit, num_augmentations, images_dir, labels_dir, output_dir, class_images)
        print(f" <-- {valid_count} total valid augmented images and labels created for '{name}'")

    print("Total for each class")
    images_output_dir = os.path.join(output_dir, "images")
    labels_output_dir = os.path.join(output_dir, "labels")
    min_size, class_images = min_class_elements(images_output_dir, labels_output_dir)

balance_dataset("backup_dataset", 'augmented_dataset', 10)