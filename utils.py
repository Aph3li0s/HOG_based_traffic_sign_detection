import cv2
import numpy as np 
def print_classes(index):
    file_path = "data/classes.txt"  
    classes = []
    with open(file_path, "r") as file:
        for line in file:
            classes.append(line.strip())
        return classes[index]

def sliding_window(box, window_size, step_size):
    x_tl, y_tl, x_br, y_br = box
    for y in range(y_tl, y_br - window_size[1] + 1, step_size):
        for x in range(x_tl, x_br - window_size[0] + 1, step_size):
            box_tl = [x, y]
            yield(x, y, [box_tl[0], box_tl[1], box_tl[0] + window_size[0], \
                        box_tl[1] + window_size[1]])
            
def split_roi(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    ratio = width / height
    sub_boxes = []
    
    if ratio > 1: # width > height
        if ratio <= 1.25:
            sub_boxes.append(list(box))
            return sub_boxes
        elif ratio <= 1.5:
            window_size = int(width / ratio)
        else:
            return None
    else: # height > width
        if ratio >= 0.8:
            sub_boxes.append(list(box))
            return sub_boxes
        elif ratio >= 0.5:
            window_size = int(height * ratio)
        else:
            if height < 100:
                return None
            else:
                window_size = width
    for (x, y, window) in sliding_window(box, (window_size, window_size), window_size//4):
        sub_boxes.append(window)
    return sub_boxes
def check_shape(image):
    thresh_square = 0.3
    height, width, _ = image.shape
    ratio = float(width) / float(height)
    # print('ratio', ratio)
    if abs(ratio - 1) < thresh_square:
        return 'square'
    else:
        if ratio > 1:
            return 'hrectangle'
        else:
            return 'vrectangle'

def split_rect(image, img_shape):
    # still 3 channels image
    if img_shape == 'square':
        # image = cv2.cvtColor(cv2.resize(image, (60, 60)), cv2.COLOR_BGR2GRAY)
        return image
    else:
        height, width, _ = image.shape

        if img_shape == 'hrectangle':
            img1 = image[:, :width // 2]
            img2 = image[:, width // 2:width]
        else:
            img1 = image[:height // 2, :]
            img2 = image[height // 2:height, :]

        # img1 = cv2.cvtColor(cv2.resize(img1, (60, 60)), cv2.COLOR_BGR2GRAY)
        # img2 = cv2.cvtColor(cv2.resize(img2, (60, 60)), cv2.COLOR_BGR2GRAY)

        return img1, img2

def normalize_channel(ch, im):
    if ch is not None:
        if ch == 0:
            nor = ((im / (np.sum(im.astype(int), axis=2, keepdims=True)\
                                + 1e-7)) * 255.0)[:, :, 0].astype(np.uint8)
            
        if ch == 2:
            nor = ((im / (np.sum(im.astype(int), axis=2, keepdims=True)\
                                + 1e-7)) * 255.0)[:, :, 2].astype(np.uint8)
            
        norm_hist = cv2.calcHist([nor], [0], None, [256], [0, 256])
        norm_hist_rev = norm_hist[::-1]
        sum_pix = np.sum(norm_hist_rev)
        allowed_pixs = [sum_pix * 0.005 if ch == 2 else sum_pix * 0.005]
        thresh = 255 - np.argmax(norm_hist_rev > allowed_pixs)
        _, thresholded = cv2.threshold(nor, thresh, 255, cv2.THRESH_BINARY)
    return thresholded
    

def transform(image):
    coor_lst = []
    roi = []
    image = cv2.resize(image, (960, 540))
    for ch in [0, 2]:
        thresholded = normalize_channel(ch, image)
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(thresholded, kernel, iterations = 1)
        dilate = cv2.dilate(erosion, kernel, iterations = 0)
        median = cv2.medianBlur(dilate, 3)
        
        contours, hierarchy = cv2.findContours(median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            if h <= 0 or w <= 0:
                continue
            
            # Extend the bounding box by 5 pixels
            x_tl = max(x - 2, 0)
            y_tl = max(y - 2, 0)
            x_br = min(x + w + 2, image.shape[1] - 1)
            y_br = min(y + h + 2, image.shape[0] - 1)
            
            # Ensure the correct ordering of coordinates
            x_tl, x_br = min(x_tl, x_br), max(x_tl, x_br)
            y_tl, y_br = min(y_tl, y_br), max(y_tl, y_br)
            
            roi.append(image[y_tl:y_br, x_tl:x_br])
            coor_lst.append((x_tl, y_tl, x_br, y_br))
    
    return roi, coor_lst

if __name__ == '__main__':
    image_name = 'test_im/test.jpg'
    from imutils.object_detection import non_max_suppression as nms
    
    im = cv2.resize(cv2.imread(image_name), (960, 540))
    roi, coor_img = transform(im)
    picked_boxes = nms(np.array(coor_img), probs=None, overlapThresh=0.3)
    print(picked_boxes)
    for (startX, startY, endX, endY) in picked_boxes:
        cv2.rectangle(im, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # for i, roi_img in enumerate(roi):
    #     cv2.imshow('a', roi_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    cv2.imshow('a', im)
    cv2.waitKey(0)