import cv2
import os
import random
import albumentations as A

class_names = []
with open('data/classes.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:        
        class_names.append(line.strip())
    # class_names.append('background')
    
def data_stat(label_root):
    class_lst = [0]*len(class_names)
    label_dir = os.listdir(label_root)
    for label in label_dir:
        label_path = os.path.join(label_root, label)
        with open(label_path, 'r') as f:
            content_list = [line.strip().split() for line in f]
            for line in content_list:
                class_lst[int(line[0])] += 1
    return class_lst

def make_augmentation(image_root, num_clone, cls):
    filenames = [filename for filename in os.listdir(image_root) if '_' + str(cls) + '.jpg' in filename]
    if filenames == []:
        return
    transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.Blur(blur_limit=(3, 5), p=1),
            A.RandomScale(scale_limit=(0.5, 1.5))
        ], p=1)
    ])
    for i in range(num_clone):
        image_filename = random.choice(filenames)
        image_path = os.path.join(image_root, image_filename)
        image = cv2.imread(image_path)
        augmented_image = transform(image=image)['image']
        # cv2.imwrite(os.path.join('bbox', f"{i}_{image_filename}"), augmented_image)  # Corrected the saving path

if __name__ == '__main__':
    class_lst = data_stat('data/labels') 
    # print(class_lst)
    # print(len(class_lst))
    # print(class_lst.index(0))
    for num_cls in class_lst:
        if num_cls < 300:
            # print(class_lst.index(num_cls))
            make_augmentation('bbox', 300-num_cls, class_lst.index(num_cls))

