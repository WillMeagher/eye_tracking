import cv2
import os
import numpy as np

import sys
sys.path.insert(0, '.')
from config import *

def get_eyes(img):
    # img = img[:,:-25]
    
    img = img[205:280, 80:-80]
    left = img[:, :150]
    right = img[:, -150:]

    left = cv2.resize(left, (30, 15), interpolation = cv2.INTER_AREA)
    right = cv2.resize(right, (30, 15), interpolation = cv2.INTER_AREA)
    
    return left, right

def main():
    # get folder to go through
    file = input("type name of filtered folder: ")
    filtered_file_path = config['path'] + "data/data/filtered/"  + file
    # check if path is valid
    if not os.path.exists(filtered_file_path + "/captures/"):
        print("invalid path")
        exit()

    open = 1 if input("open eyes? (y/n): ") == 'y' else 0

    # get ouptut file path
    output_file_path = config['path'] + "data/data/filtered/" + file + "_eyes/"
    # make output if not exists
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    images = os.listdir(filtered_file_path + "/captures/")
    i = 0

    while i < len(images):
        img = cv2.imread(filtered_file_path + "/captures/" + images[i])

        left, right = get_eyes(img)

        cv2.imwrite(output_file_path + str(open) + "_left_" + images[i], left)
        cv2.imwrite(output_file_path + str(open) + "_right_" + images[i], right)

        i += 1

if __name__ == "__main__":
    main()