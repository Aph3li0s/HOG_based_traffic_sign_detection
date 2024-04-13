import os
import cv2
import numpy as np
import random
def get_box(image, label_dir):
    boxes = []
    cropped_images =[]
    class_labels = []  # Initialize class label list
    with open(label_dir, 'r') as f:
        file = f.readlines()
        if file:  # Check if the file is not empty
            for line in file:  # Iterate through each line in the file
                parts = line.split()
                class_label = int(parts[0])  # Extract class label
                class_labels.append(class_label)  # Append class label to the list
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                #get box
                box = (x_center, y_center, width, height)
                #get coordinates
                left = int((box[0] - box[2] / 2) * image.shape[1])
                right = int((box[0] + box[2] / 2) * image.shape[1])
                top = int((box[1] - box[3] / 2) * image.shape[0])
                bottom = int((box[1] + box[3] / 2) * image.shape[0])
                #append box and cropped_image
                boxes.append([left, right, top, bottom])
                cropped_images.append(image[top:bottom, left:right])
    return class_labels, boxes, cropped_images

def crop_n_pieces_except_bbox(image, bbox_list, n):
    height, width, _ = image.shape
    pieces = []
    
    bbox_mask = np.zeros((height, width), dtype=bool)
    for bbox in bbox_list:
        left, right, top, bottom = bbox
        bbox_mask[top:bottom, left:right] = True
        
    while len(pieces) < n:
        piece_size = random.randint(40, 120)
        x = np.random.randint(0, width - piece_size + 1)
        y = np.random.randint(height//2, height - piece_size + 1)

        if not np.any(bbox_mask[y:y + piece_size, x:x + piece_size]):
            pieces.append(image[y:y + piece_size, x:x + piece_size])

    return pieces


def cropBox(images_dir, labels_dir):
    """
    Crop bounding box from images.
    """
    list_items = os.listdir(images_dir)

    for item in list_items: 
        name_file = item.replace('.jpg', '')
        image_path = os.path.join(images_dir, item)
        image_root = cv2.imread(image_path)
        label_name = item.replace('jpg', 'txt')
        label_path = os.path.join(labels_dir, label_name)
        class_label, boxes, cropped_images = get_box(image_root, label_path)

        for idx, cropped_image in enumerate(cropped_images):
            save_path = os.path.join('bbox', f"{name_file}_{class_label[idx]}.jpg")
            cv2.imwrite(save_path, cropped_image)
            print(f"Saved cropped image: {save_path}")

        
    # cv2.imshow('image', cropped_images[0])
    # for idx, image in enumerate(cropped_images):
    #     resized_image = cv2.resize(image, (120, 120))
    #     cv2.imshow(f'Image {idx}', resized_image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
def cropBgr(images_dir, labels_dir):
    """
    Crop images into n pieces except for the bounding box.
    """
    list_items = os.listdir(images_dir)

    for item in list_items: 
        name_file = item.replace('.jpg', '')
        image_path = os.path.join(images_dir, item)
        image_root = cv2.imread(image_path)
        label_name = item.replace('jpg', 'txt')
        label_path = os.path.join(labels_dir, label_name)
        class_label, boxes, cropped_images = get_box(image_root, label_path)
        
        a = crop_n_pieces_except_bbox(image_root, boxes, 5)
        for idx, cropped_image in enumerate(a):
            save_path = os.path.join('bbox2', f"{name_file}_{idx}.jpg")

            cv2.imwrite(save_path, cropped_image)
            print(f"Saved cropped image: {item}")
if __name__ == "__main__":
    folder_path = 'bbox'
    [os.remove(os.path.join(folder_path, file)) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    # folder_path = 'bbox2'
    # [os.remove(os.path.join(folder_path, file)) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    cropped_data = cropBox('data/images', 'data/labels')
    