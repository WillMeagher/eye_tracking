import cv2
import os
import numpy as np
from get_eyes import get_eyes

import sys
sys.path.insert(0, '.')
from config import *

# get folder to go through
file = input("type name of filtered folder: ")

# get full file path
filtered_file_path = config['path'] + "data/data/filtered/" + file

# get ouptut file path
file = input("type name of file to put images: ")
processed_file_path = config['path'] + "data/data/" + file + "/"

# check if input path is valid
if not os.path.exists(filtered_file_path + "/captures/"):
    print("invalid path")
    exit()

# map image data to dictionary
image_data = os.listdir(filtered_file_path + "/captures/")

# check if output path is valid
if not os.path.exists(processed_file_path):
    print("invalid path")
    exit()

# map orientation data to dictionary
orientation_data = []
orientation_file = open(filtered_file_path + '/orientation_data.txt', 'r')
for line in orientation_file.readlines():
    orientation_data.append(line.strip())
orientation_file.close()

img_i = 0
ori_i = 0
min_time = 1000000

while img_i < len(image_data) - 1 and ori_i < len(orientation_data):
    temp_time = int(image_data[img_i].split(".")[0]) - int(orientation_data[ori_i].split("_")[0])

    closer = abs(temp_time) < min_time
    ori_is_before = temp_time > 0
    close_enough = abs(temp_time) <= 250
    
    if close_enough:
        if ori_is_before:
            min_time = abs(temp_time)
            ori_i += 1
            continue
        else:
            if not closer:
                ori_i -= 1
    else:
        if ori_is_before:
            ori_i += 1
            continue
        else:
            ori_i -= 1
            img_i += 1
            min_time = 1000000
            continue

    img = cv2.imread(filtered_file_path + "/captures/" + image_data[img_i])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # crop out eyes
    # img = img[:,:-25]
    left, right = get_eyes(img)
    eyes = np.concatenate((left, right), axis=1)

    eyes = cv2.resize(eyes, (0,0), fx=0.2, fy=0.2)
    
    # cv2.imshow('eyes', eyes)
    # cv2.waitKey(0)

    # save image to processed data
    cv2.imwrite(processed_file_path + image_data[img_i].split(".")[0] + "_" + orientation_data[ori_i] + ".jpg", eyes)

    min_time = 1000000
    img_i += 1
    ori_i -= 1
