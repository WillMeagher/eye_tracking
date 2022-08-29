import numpy as np
import os

import sys
sys.path.insert(0, '.')
from config import *

# get folder to go through
data_file = input("type name of data folder: ")
data_file_path = config['path'] + "data/data/final/" + data_file
# check if input path is valid
if not os.path.exists(data_file_path):
    print("invalid path")
    exit()


def import_data(type="train"):
    features = []
    labels = []

    for file_name in os.listdir(data_file_path):
        if file_name.split("_")[0] == type:
            training_data = np.load(data_file_path + "/" + file_name)
            features.extend(np.array(training_data['a']))
            labels.extend(np.array(training_data['b']))

    return np.array(features), np.array(labels)


def main():
    picture_data, label_data = import_data()
    print(picture_data, label_data)

if __name__ == "__main__":
    main()