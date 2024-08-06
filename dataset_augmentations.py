import os
import cv2
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

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

# Define augmentation pipeline
seq = iaa.Sequential([
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.25)),
    # n of the pixels in an image with salt and pepper noise
    iaa.SaltAndPepper(0.01),
    # make some images brighter or darker in 20% of all cases
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Scale/zoom them, translate/move them and rotate them
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
    )
], random_order=True)

image_dir = './backup_dataset/images'
label_dir = './backup_dataset/labels'
output_dir = './augmented_dataset'

os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

# Number of augmentations per image
num_augmentations = 10
valid_count = 0

for img_file in os.listdir(image_dir):
    if img_file.endswith(('.jpg', '.png', '.jpeg')):
        # Read image
        img_path = os.path.join(image_dir, img_file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read corresponding label file
        label_file = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
        with open(label_file, 'r') as f:
            yolo_annotation = f.read().strip()

        # Convert YOLO annotation to bounding box format
        bbox = yolo_to_bbox(yolo_annotation, image.shape[1], image.shape[0])
        bbs = BoundingBoxesOnImage([bbox], shape=image.shape)

        # Apply augmentations
        for i in range(num_augmentations):
            aug_image, aug_bbs = seq(image=image, bounding_boxes=bbs)
            
            for bbox in aug_bbs.bounding_boxes:
                # Check if the bounding box is valid in reference of the size of the image
                if (0 <= bbox.x1 < aug_image.shape[1] and 0 <= bbox.x2 < aug_image.shape[1]
                    and 0 <= bbox.y1 < aug_image.shape[0] and 0 <= bbox.y2 < aug_image.shape[0]):
                    
                    # Generate filenames
                    aug_img_file = f"{os.path.splitext(img_file)[0]}_aug{i+1}.jpg"
                    aug_label_file = f"{os.path.splitext(img_file)[0]}_aug{i+1}.txt"
                    
                    # Save augmented image
                    cv2.imwrite(os.path.join(output_dir, 'images', aug_img_file),
                                cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                    
                    # Save label file
                    with open(os.path.join(output_dir, 'labels', aug_label_file), 'w') as f:
                        yolo_ann = bbox_to_yolo(bbox, aug_image.shape[1], aug_image.shape[0])
                        f.write(yolo_ann + '\n')
                    
                    valid_count += 1

print("Augmentation completed.")
print(f"{valid_count} valid augmented images and labels created")