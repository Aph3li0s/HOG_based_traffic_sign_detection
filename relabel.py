import os
import cv2
import numpy as np

""" 
    This use for adjustment of the label files (from 50 classes to 45 classes)
"""
with open('v2/data/classes.txt', 'r') as file:
    label = []
    for line in file:
        label.append(line.strip())

label_dir = os.listdir('v1/data/labels')
for label_file in label_dir:
    line = []
    with open(f'v1/data/labels/{label_file}', 'r') as file:
        content_list = [line.strip().split() for line in file]

    filtered_content = []
    for line in content_list:
        if int(line[0]) >= 25:
            if int(line[0]) == 25 or int(line[0]) == 26 or int(line[0]) == 27 or int(line[0]) == 29 or int(line[0]) == 31 or int(line[0]) == 32:
                filtered_content.append(line)
                # print(label_file)
            else:
                if int(line[0]) == 28:  
                    line[0] = '25'
                    
                elif int(line[0]) == 30:
                    line[0] = '26'

                    
                else:
                    line[0] = str(int(line[0]) - 6)
                    # print(line)

    with open(f'v2/data/labels/{label_file}', 'w') as file:
        for line in content_list:
            if line not in filtered_content:
                file.write(' '.join(line) + '\n')
