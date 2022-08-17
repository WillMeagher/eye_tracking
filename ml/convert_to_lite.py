import tensorflow as tf
import os

import sys
sys.path.insert(0, '.')
from config import *

file_folder = input("type ml folder that model is in: ")

in_file = input("type name of model to convert: ")
saved_model_dir = config['path'] + 'ml/' + file_folder + '/models/' + in_file

# check if input path is valid
if not os.path.exists(saved_model_dir):
    print("invalid path")
    exit()

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir, signature_keys=['serving_default'])

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()

out_file = input("type what to name the file: ")

fo = open(config['path'] + 'ml/' + file_folder + '/models/lite_models/' + out_file + '.tflite', "wb")
fo.write(tflite_model)
fo.close