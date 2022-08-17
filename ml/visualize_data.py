import matplotlib.pyplot as plt
import pickle

import sys
sys.path.insert(0, '.')
from config import *

data_path = config['path'] + "data/final"

feature_data = []
label_data = []

x = []
y = []

with open(data_path + "/features.txt",'rb') as file_object:
    feature_data = pickle.load(file_object)

with open(data_path + "/labels.txt",'rb') as file_object:
    label_data = pickle.load(file_object)


for label in label_data:
    x.append(label[0])
    y.append(label[1])

plt.scatter(x, y)
plt.show()
