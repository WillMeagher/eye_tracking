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

epochs = 5
iterations = 2000
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def distance_between(v1, v2):
    return math.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)


def evaluate(m):
    predictions = m.predict(X_test)
    results = []
    for i in range(len(predictions)):
        distance = distance_between(predictions[i], Y_test[i]) * 90
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


(X_train, Y_train) = import_data.import_data_train()
(X_test, Y_test) = import_data.import_data_test()

X_train = np.array(X_train).astype('float32')
Y_train = np.array(Y_train).astype('float32')
X_test = np.array(X_test).astype('float32')
Y_test = np.array(Y_test).astype('float32')

split_nums = 1

while len(X_train) / split_nums > 300000:
    split_nums += 1

X_train_list = np.array_split(X_train, split_nums)
Y_train_list = np.array_split(Y_train, split_nums)

for i in range(iterations):

    model = get_model()

    for i in range(epochs):
        print("Epoch " + str(i + 1))
        for j in range(len(X_train_list)):
            model.fit(X_train_list[j], Y_train_list[j], epochs=1, batch_size=256)
            gc.collect()
        evaluation = evaluate(model)

        # exit if model isnt learning
        if evaluation > 30:
            break

    model.summary()

    evaluation = evaluate(model)

    if evaluation < 4:
        model.save(config['path'] + 'ml/models/' + str(int(time.time() * 1000)) + '_' + str(format(evaluation,".5f")) + '_model')

    del model
    gc.collect()
