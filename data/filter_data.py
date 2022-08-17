import cv2
import os
import numpy as np
import distortion

import sys
sys.path.insert(0, '.')
from config import *

def update_orientation_data(orientation_file):
    # map orientation data to dictionary
    orientation_data = []
    for line in orientation_file.readlines():
        data = line.split("_")
        orientation_data.append([int(data[0]) ,[float(data[1]), float(data[2])]])
    orientation_file.close()

    final_data = []

    for i in range(len(orientation_data)):
        if i + 4 < len(orientation_data):
            if orientation_data[i+4][0] - orientation_data[i][0] < 500:
                time_data = [orientation_data[i][0], orientation_data[i+1][0], orientation_data[i+2][0], orientation_data[i+3][0], orientation_data[i+4][0]]
                x_data = [orientation_data[i][1][0], orientation_data[i+1][1][0], orientation_data[i+2][1][0], orientation_data[i+3][1][0], orientation_data[i+4][1][0]]
                y_data = [orientation_data[i][1][1], orientation_data[i+1][1][1], orientation_data[i+2][1][1], orientation_data[i+3][1][1], orientation_data[i+4][1][1]]
                if max(x_data) - min(x_data) < 3 and max(y_data) - min(y_data) < 3 and max(abs(np.array(y_data))) < 45 and max(abs(np.array(x_data))) < 45:
                    final_data.append(str(int(np.mean(time_data))) + "_" + format(np.mean(x_data), '.5f') + "_" + format(np.mean(y_data), '.3f'))

    return final_data


# get folder to go through
file = input("type name of raw folder: ")

raw_file_path = config['path'] + "data/raw/"  + file
filtered_file_path = config['path'] + "data/filtered/" + file

# check if path is valid
if not os.path.exists(raw_file_path + "/captures/"):
    print("invalid path")
    exit()

# copy over files if it hasnt been done already
if not os.path.exists(filtered_file_path + "/captures/"):
    os.makedirs(filtered_file_path + "/captures/")
    old = open(raw_file_path + '/orientation_data.txt', 'r')
    new = open(filtered_file_path + '/orientation_data.txt', 'w')

    # update and filter orientation data
    new_ori_data = update_orientation_data(old)

    for line in new_ori_data:
        new.write(line + "\n")

    old.close()
    new.close()

images = os.listdir(raw_file_path + "/captures/")
i = 0

while i < len(images):
    # print(raw_file_path + "/captures/" + images[i])
    img = cv2.imread(raw_file_path + "/captures/" + images[i])
    # small = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 

    cv2.imshow('img', img)

    key = cv2.waitKey(0)

    if key == 113 and i > 0:
        i -= 1
    elif key == 119:
        if os.path.exists(filtered_file_path + "/captures/" + images[i]):
            os.remove(filtered_file_path + "/captures/" + images[i])
        i += 1
    elif key == 101:
        if not os.path.exists(filtered_file_path + "/captures/" + images[i]):
            cv2.imwrite(filtered_file_path + "/captures/" + images[i], img)
        i += 1
    else:
        continue