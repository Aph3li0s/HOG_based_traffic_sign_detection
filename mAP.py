from mapcalc import calculate_map, calculate_map_range
import os
import cv2
import numpy as np

def read_txt(label_root):
    with open(label_root, 'r') as file:
        content_list = [line.strip().split() for line in file]
    return content_list

def read_gt(data_root):
    images = os.listdir(os.path.join(data_root, 'images'))
    labels = os.listdir(os.path.join(data_root, 'labels'))  
    
    for idx in range(len(images)):
        label = read_txt(os.path.join(data_root, 'labels', labels[idx]))
        
def calculate_mAP(gt_boxes, gt_labels, result_boxes, result_labels, iou_thresh):
    ground_truth = {
    'boxes': gt_boxes,

    'labels': gt_labels,
    }

    result_dict = {
        'boxes': result_boxes,
        'labels': result_labels,
    }
    # print(ground_truth)
    # print(result_dict)
    return calculate_map(ground_truth, result_dict, iou_thresh)

if __name__ == '__main__':
    pass
    read_gt('v2/data')