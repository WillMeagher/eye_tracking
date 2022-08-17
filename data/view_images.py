import os
import cv2

import sys
sys.path.insert(0, '.')
from config import *

file = input("type name of img folder in data: ")
file_path = config['path'] + "data/"  + file

# check if path is valid
if not os.path.exists(file_path):
    print("invalid path")
    exit()

images = os.listdir(file_path)
i = 0

while i < len(images):
    img = cv2.imread(file_path + images[i])

    cv2.imshow('img', img)

    key = cv2.waitKey(0)

    if key == 113 and i > 0:
        i -= 1
    elif key == 119:
        i += 1
    elif key == 101:
        i += 1
    else:
        continue