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

def detect_quick(image_root):
    # width, height = 960, 540
    t0 = time.time()
    image = cv2.imread(image_root)
    height, width = image.shape[:2]
    # image = cv2.resize(image, ((width*3)//4, (height*3//4)))
    # image = cv2.resize(image, (width//2, height//2))
    # image = cv2.resize(cv2.imread(image_root), (width, height))
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
    print(des_label)
    color_lst = random_colors()
    # print(label)
    # print(final_boxes)
    for idx in final_boxes:
        for box in idx:
            plot_bbox_labels(image, box, des_label[final_boxes.index(idx)], color_lst[final_boxes.index(idx)], 0.4)
    print("Time taken: ", time.time() - t0)
    cv2.imshow('image', cv2.resize(image, (image.shape[1]//2, image.shape[0]//2)))
    cv2.imwrite('result.jpg', image)
    cv2.waitKey(0)

if __name__ == '__main__':

    # print(detect_quick('test_data/0592.jpg'))
    detect_quick('test_im/g.jpg')

    # print(random_colors())
    



