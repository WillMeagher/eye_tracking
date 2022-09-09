import numpy as np
import gc

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def train_model(model, evaluate_function, X_train, Y_train, X_test, Y_test, epochs=5, min_continue=30):

    X_train = np.array(X_train).astype('float32')
    Y_train = np.array(Y_train).astype('float32')
    X_test = np.array(X_test).astype('float32')
    Y_test = np.array(Y_test).astype('float32')

    # break up data
    split_nums = 1
    while len(X_train) / split_nums > 300000:
        split_nums += 1

    X_train_list = np.array_split(X_train, split_nums)
    Y_train_list = np.array_split(Y_train, split_nums)

    # train model
    for i in range(epochs):
        print("Epoch " + str(i + 1))
        for j in range(len(X_train_list)):
            model.fit(X_train_list[j], Y_train_list[j], epochs=1, batch_size=256)
            gc.collect()

        # exit if model isnt learning
        evaluation = evaluate_function(model, X_test, Y_test)
        if evaluation > min_continue:
            break
        evaluate_function(model, X_train[:5000], Y_train[:5000])

    return model, evaluation
    