import numpy as np

import sys
sys.path.insert(0, '.')
from config import *

data_path = config['path'] + "data/final"
data_path_2 = config['path'] + "data/final_2"

def import_data_train():
    training_data = np.load(data_path + '/train.npz')

    return np.array(training_data['a']), np.array(training_data['b'])

def import_data_test():
    testing_data = np.load(data_path_2 + '/test.npz')

    return np.array(testing_data['a']), np.array(testing_data['b'])

def main():
    picture_data, label_data = import_data_train()
    print(picture_data, label_data)

if __name__ == "__main__":
    main()