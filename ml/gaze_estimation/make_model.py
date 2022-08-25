from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D
import import_data
import time
import math
import gc

import sys
sys.path.insert(0, '.')
from config import *

sys.path.insert(0, './ml')
from train_model import train_model

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def evaluate(model, X, Y):
    def distance_between(v1, v2):
        return math.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)

    predictions = model.predict(X)
    results = []
    for i in range(len(predictions)):
        distance = distance_between(predictions[i], Y[i]) * 90
        results.append(distance)

    results = np.array(results)

    evaluation = np.mean(results)
    print(evaluation)
    return evaluation


def get_model():
    model = keras.Sequential()
    model.add(Conv2D(1, (3, 3), input_shape=(15, 60, 1)))

    model.add(Flatten())

    model.add(Dense(500, activation= "relu"))
    model.add(Dropout(.1))

    model.add(Dense(500, activation= "relu"))
    model.add(Dropout(.1))

    model.add(Dense(100, activation= "relu"))

    model.add(Dense(2, activation= "relu"))

    model.compile(optimizer='adam',
                loss="mean_squared_error",
                metrics=["mean_squared_error"])

    return model


def main():
    model_shape = get_model()
    epochs = 5
    min_continue = 30
    min_save = 3
    this_file_path = 'ml/gaze_estimation/models/'

    (X_train, Y_train) = import_data.import_data('train')
    (X_test, Y_test) = import_data.import_data('test')

    while True:
        new_model, eval = train_model(model_shape, evaluate, X_train, Y_train, X_test, Y_test, epochs, min_continue)
        
        if eval < min_save:
            new_model.save(config['path'] + this_file_path + str(int(time.time() * 1000)) + '_' + str(format(eval,".5f")) + '_model')

        del new_model
        gc.collect()


if __name__ == '__main__':
    main()