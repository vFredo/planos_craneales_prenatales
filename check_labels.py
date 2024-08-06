import cv2
import os
import numpy as np


def plot_one_box(x, img, color, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_labels_on_image(image_path, label_path, output_path, class_names):
    # Read image
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Read YOLO format labels
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split()
        class_id = int(data[0])
        x_center, y_center, w, h = map(float, data[1:])

        # Convert YOLO format to pixel coordinates
        x1 = int((x_center - w/2) * width)
        y1 = int((y_center - h/2) * height)
        x2 = int((x_center + w/2) * width)
        y2 = int((y_center + h/2) * height)

        # Draw bounding box
        label = class_names[class_id]
        color = COLOR_MAP[label]
        plot_one_box([x1, y1, x2, y2], img, color=color, label=label)

    # Save the image with drawn bounding boxes
    cv2.imwrite(output_path, img)

image_folder = 'augmented_dataset/images'
label_folder = 'augmented_dataset/labels'
output_folder = 'label_correctness'
class_names = ["trans_cerebellum", "trans_thalamic", "trans_ventricular"]
# Define a color map for each class
COLOR_MAP = {
    "trans_cerebellum": (255, 0, 0),
    "trans_thalamic": (0, 255, 0),
    "trans_ventricular": (0, 0, 255)
}

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)
        label_path = os.path.join(label_folder, filename.rsplit('.', 1)[0] + '.txt')
        output_path = os.path.join(output_folder, filename)

        if os.path.exists(label_path):
            draw_labels_on_image(image_path, label_path, output_path, class_names)
            print(f"Processed: {filename}")
        else:
            print(f"Label file not found for: {filename}")
