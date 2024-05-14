import cv2
import os
import numpy as np
from utils import print_classes
from mAP import read_txt
import time

def random_colors():
    file_path = "data/classes.txt"  
    num_classes = []
    with open(file_path, "r") as file:
        for line in file:
            num_classes.append(line.strip())
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                    np.random.randint(255),
                    np.random.randint(255)) for _ in range(len(num_classes))]
    return class_colors

def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img

def detect_quick(img_dir):
    image = cv2.imread(os.path.join('data/images',img_dir))
    height, width = image.shape[:2]
    label_dir = os.path.join('data/labels', img_dir.replace('jpg', 'txt'))
    txt_label = read_txt(label_dir)
    gt_label = [int(label[0]) for label in txt_label]
    gt_boxes = [box[1:] for box in txt_label]
    
    gt_boxes2 = [
    [
        int((float(obj_box[0])-(float(obj_box[2])/2)) * width),  # x_tl
        int((float(obj_box[1]) - (float(obj_box[3])/2)) * height),  # y_tl
        int((float(obj_box[0]) + (float(obj_box[2])/2)) * width),  # x_br
        int((float(obj_box[1]) + (float(obj_box[3])/2)) * height)  # y_br
    ] 
    for obj_box in gt_boxes
    ]
    gt_boxes3 = [[(x1, y1, x2, y2)] for x1, y1, x2, y2 in gt_boxes2]
    des_label = [print_classes(i) for i in gt_label]

    print('box; ', gt_boxes3)
    print('label: ', gt_label)
    print(des_label)
    color_lst = random_colors()
    for idx in gt_boxes3:
        for box in idx:
            plot_bbox_labels(image, box, des_label[gt_boxes3.index(idx)], color_lst[gt_boxes3.index(idx)], 0.4)
    cv2.imwrite('result.jpg', image)
    # cv2.waitKey(0)

if __name__ == '__main__':
    # print(detect_w_label('3110.jpg'))
    t0 = time.time()
    print(detect_quick('0592.jpg'))
    print("Time taken: ", time.time() - t0)
    # print(random_colors())
    



