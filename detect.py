import cv2
import os
import numpy as np
from imutils.object_detection import non_max_suppression as nms
import joblib
from skimage.feature import hog
from utils import transform, split_roi, print_classes
from mAP import read_txt, calculate_mAP
import time

def detect(roi_img):
    pca_model = joblib.load('model/pca_1000.pkl')
    lda_model = joblib.load('model/lda_pca_1000.pkl')
    features = hog(roi_img, pixels_per_cell=(4, 4), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")
    features = np.array(features).reshape(1, -1)
    pca_features = pca_model.transform(features)
    predicted_class = lda_model.predict(pca_features)
    
    if predicted_class == 'background':
        return 'background'
    else:
        return int(predicted_class[0])


def detect_w_label(image_root):
    data_dir = os.path.join('data/images', image_root)
    label_dir = os.path.join('data/labels', image_root.replace('jpg', 'txt'))
    width, height = 960, 540
    image = cv2.resize(cv2.imread(data_dir), (width, height))

    roi_img, coor_img = transform(image)
    picked_boxes = nms(np.array(coor_img), probs=None, overlapThresh=0.3)

    final_boxes, label = [], []
    for i, box in enumerate(picked_boxes):
        x_tl, y_tl, x_br, y_br = box
        sub_boxes = split_roi(box)
        if sub_boxes is not None:
            if len(sub_boxes) == 1: # square
                x_tl, y_tl, x_br, y_br = sub_boxes[0]
                detect_img = image[y_tl:y_br, x_tl:x_br]
                detect_img = cv2.cvtColor(cv2.resize(detect_img, (64, 64)), cv2.COLOR_BGR2GRAY)
                pred_label = detect(detect_img)
                if pred_label != 'background':
                    label.append(pred_label)
                    final_boxes.append([tuple(box)])
            else: # rectangle
                for box in sub_boxes:
                    x_tl, y_tl, x_br, y_br = box
                    detect_img = image[y_tl:y_br, x_tl:x_br]
                    detect_img = cv2.cvtColor(cv2.resize(detect_img, (64, 64)), cv2.COLOR_BGR2GRAY)
                    pred_label = detect(detect_img)

                    if pred_label != 'background':
                        label.append(pred_label)
                        final_boxes.append([tuple(box)])
                        continue

    txt_label = read_txt(label_dir)
    gt_label = [int(label[0]) for label in txt_label]
    gt_boxes = [box[1:] for box in txt_label]
    
    final_boxes2 =  [[i for i in lst[0]] for lst in final_boxes]
    gt_boxes2 = [
    [
        int((float(obj_box[0])-(float(obj_box[2])/2)) * width),  # x_tl
        int((float(obj_box[1]) - (float(obj_box[3])/2)) * height),  # y_tl
        int((float(obj_box[0]) + (float(obj_box[2])/2)) * width),  # x_br
        int((float(obj_box[1]) + (float(obj_box[3])/2)) * height)  # y_br
    ] 
    for obj_box in gt_boxes
    ]

    des_label = [print_classes(i) for i in label]
    des_gt_label = [print_classes(i) for i in gt_label]
    for idx in final_boxes:
        for box in idx:
            x_tl, y_tl, x_br, y_br = box
            cv2.rectangle(image, (x_tl, y_tl), (x_br, y_br), (0, 255, 0), 1)
            
            caption = des_label[final_boxes.index(idx)]
            text_size, _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x_tl
            text_y = y_tl + text_size[1] + 5  # Position text below the top-left corner of the box
            cv2.putText(image, caption, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    gt_boxes3 = [[(x1, y1, x2, y2)] for x1, y1, x2, y2 in gt_boxes2]
    # print(gt_boxes3)
    # print(final_boxes)
    for idx in gt_boxes3:
        for box in idx:
            x_tl, y_tl, x_br, y_br = box
            cv2.rectangle(image, (x_tl, y_tl), (x_br, y_br), (255, 255, 0), 1)
            
            caption = des_gt_label[gt_boxes3.index(idx)]
            text_size, _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x_tl
            text_y = y_tl + text_size[1] + 5  # Position text below the top-left corner of the box
            cv2.putText(image, caption, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # print("box: ", gt_boxes2, "\t", final_boxes)
    # print("label: ", gt_label, "\t", label)
    cv2.imshow('image', cv2.resize(image, (960, 540)))
    cv2.waitKey(0)
    a = calculate_mAP(gt_boxes2, gt_label, final_boxes2, label, 0.5)
    return a

def detect_quick(image_root):
    width, height = 960, 540
    image = cv2.resize(cv2.imread(image_root), (width, height))
    roi_img, coor_img = transform(image)
    picked_boxes = nms(np.array(coor_img), probs=None, overlapThresh=0.3)

    final_boxes, label = [], []
    for i, box in enumerate(picked_boxes):
        x_tl, y_tl, x_br, y_br = box
        sub_boxes = split_roi(box)
        if sub_boxes is not None:
            if len(sub_boxes) == 1: # square
                x_tl, y_tl, x_br, y_br = sub_boxes[0]
                detect_img = image[y_tl:y_br, x_tl:x_br]
                detect_img = cv2.cvtColor(cv2.resize(detect_img, (64, 64)), cv2.COLOR_BGR2GRAY)
                pred_label = detect(detect_img)
                if pred_label != 'background':
                    label.append(pred_label)
                    final_boxes.append([tuple(box)])
            else: # rectangle
                for box in sub_boxes:
                    x_tl, y_tl, x_br, y_br = box
                    detect_img = image[y_tl:y_br, x_tl:x_br]
                    detect_img = cv2.cvtColor(cv2.resize(detect_img, (64, 64)), cv2.COLOR_BGR2GRAY)
                    pred_label = detect(detect_img)

                    if pred_label != 'background':
                        label.append(pred_label)
                        final_boxes.append([tuple(box)])
                        continue
    des_label = [print_classes(i) for i in label]
    for idx in final_boxes:
        for box in idx:
            x_tl, y_tl, x_br, y_br = box
            cv2.rectangle(image, (x_tl, y_tl), (x_br, y_br), (0, 255, 0), 1)
            
            caption = des_label[final_boxes.index(idx)]
            text_size, _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x_tl
            text_y = y_tl + text_size[1] + 5  # Position text below the top-left corner of the box
            cv2.putText(image, caption, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
    cv2.imshow('image', cv2.resize(image, (960, 540)))
    cv2.waitKey(0)

if __name__ == '__main__':
    # print(detect_w_label('1643.jpg'))
    print(detect_quick('data/images/1643.jpg'))
    



