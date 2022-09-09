from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D
import time
import cv2
import gc

import sys
sys.path.insert(0, '.')
from config import *

sys.path.insert(0, './ml')
from train_model import train_model
from import_data import import_data

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def evaluate(model, X, Y):
    predictions = model.predict(X)
    results = []

    for i in range(len(predictions)):
        if round(predictions[i][0]) == Y[i]:
            results.append(1)
        else:
            results.append(0)

    results = np.array(results)

    evaluation = 1 - np.mean(results)
    print(evaluation)
    return evaluation


def evaluate_and_show(model, X, Y):
    predictions = model.predict(X)
    results = []

    for i in range(len(predictions)):
        if round(predictions[i][0]) == Y[i]:
            results.append(1)
        else:
            img = cv2.resize(X[i], (0,0), fx=10, fy=10)
            cv2.imshow('image ' + str(round(predictions[i][0])),img)
            cv2.waitKey(0)
            results.append(0)

    results = np.array(results)

    evaluation = 1 - np.mean(results)
    print(evaluation)
    return evaluation


def get_model():
    model = keras.Sequential()
    model.add(Conv2D(4, (4, 4), input_shape=(15, 30, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(8, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.5))

    model.add(Flatten())

    model.add(Dense(300, activation= "relu"))
    model.add(Dropout(.3))

    model.add(Dense(100, activation= "relu"))
    model.add(Dropout(.3))

    model.add(Dense(1, activation= "sigmoid"))

    model.compile(optimizer='adam',
                loss="binary_crossentropy",
                metrics=["accuracy"])

    return model


def main():
    epochs = 5
    min_continue = .2
    min_save = .0015
    this_file_path = 'ml/eye_status/models/'

    (X_train, Y_train) = import_data('train')
    (X_test, Y_test) = import_data('test')

    while True:
        model_shape = get_model()
        new_model, eval = train_model(model_shape, evaluate, X_train, Y_train, X_test, Y_test, epochs, min_continue)
        
        if eval < min_save:
            new_model.save(config['path'] + this_file_path + str(int(time.time() * 1000)) + '_' + str(format(eval,".5f")) + '_model')

        # evaluate_and_show(new_model, X_test, Y_test)

        del new_model
        gc.collect()


if __name__ == '__main__':
    main()