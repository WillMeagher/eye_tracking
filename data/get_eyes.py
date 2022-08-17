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

    return left, right

def main():
    # get folder to go through
    file = input("type name of raw folder: ")

    raw_file_path = config['path'] + "data/raw/"  + file
    eyes_file_path = config['path'] + "data/eyes/"

    # check if path is valid
    if not os.path.exists(raw_file_path + "/captures/"):
        print("invalid path")
        exit()

    images = os.listdir(raw_file_path + "/captures/")
    i = 0

    while i < len(images):
        img = cv2.imread(raw_file_path + "/captures/" + images[i])

        left, right = get_eyes(img)

        print(right.shape)
        exit()

        cv2.imwrite(eyes_file_path + "left_" + images[i], left)
        cv2.imwrite(eyes_file_path + "right_" + images[i], right)

        i += 1

if __name__ == "__main__":
    main()