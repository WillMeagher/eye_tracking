import tensorflow
import import_data
import numpy as np
import math
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '.')
from config import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def distance_between(v1, v2):
    return math.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)

model_paths = config['path'] + 'ml/models/'

models =["1657843229563_2.56348_model"] #os.listdir(model_paths)

for model_path in models:

    model = tensorflow.keras.models.load_model(model_paths + model_path)

    (X_data, Y_data) = import_data.import_data_test()

    X_data = np.array(X_data).astype('float32')
    Y_data = np.array(Y_data).astype('float32')

    predictions = model.predict(X_data)

    results = []
    results_x = []
    results_y = []

    for i in range(len(predictions)):
        distance = distance_between(predictions[i], Y_data[i]) * 90
        results_x.append((predictions[i][0] - Y_data[i][0]) * 90)
        results_y.append((predictions[i][1] - Y_data[i][1]) * 90)
        results.append(distance)

    results = np.array(results)

    actual = (Y_data * 90) - 45
    actual_x = list(map(lambda x: x[0], actual))
    actual_y = list(map(lambda x: x[1], actual))


    plt.scatter(actual_x, results_x)
    plt.show()

    plt.scatter(actual_y, results_y)
    plt.show()

    print(np.mean(results))