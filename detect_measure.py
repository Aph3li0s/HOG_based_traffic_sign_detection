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


def read_test(image_root):
    data_dir = os.path.join('data/images', image_root)
    label_dir = os.path.join('data/labels', image_root.replace('jpg', 'txt'))
    width, height = 960, 540
    image = cv2.resize(cv2.imread(data_dir), (width, height))
    for f in os.listdir('clipped_roi'):
        os.remove(f'clipped_roi/{f}')
    roi_img, coor_img = transform(image)
    picked_boxes = nms(np.array(coor_img), probs=None, overlapThresh=0.3)

    final_boxes, label = [], []
    for i, box in enumerate(picked_boxes):
        x_tl, y_tl, x_br, y_br = box
        cropped_img = image[y_tl:y_br, x_tl:x_br]
        # cv2.imwrite(f'clipped_roi/{i}.jpg', cropped_img)
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
                    # cv2.imwrite(f'clipped_roi/{i}_{j}.jpg', detect_img)
                    # j += 1
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
    mAP = calculate_mAP(gt_boxes2, gt_label, final_boxes2, label, 0.5)
    return mAP

if __name__ == '__main__':
    test_dir = os.listdir('test_data')
    lst = []
    for f in test_dir:
        result = read_test(f)
        print(f, result)
        lst.append(result)
    print(sum(lst)/len(lst))
    



