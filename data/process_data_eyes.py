import cv2
import os
import numpy as np
from get_eyes import get_eyes
import shutil

import sys
sys.path.insert(0, '.')
from config import *

TEMP_FILE_PATH = config['path'] + "data/data/temp/"


def process(filtered_file_path):
    feature_data = []
    label_data = []

    image_data = os.listdir(filtered_file_path)

    for img_file in image_data:
        # get image
        img = cv2.imread(filtered_file_path + "/" + img_file)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
        img = cv2.resize(img, (30, 30), interpolation = cv2.INTER_AREA)

        # get label
        label = int(img_file.split("_")[0])

        feature_data.append(img)
        label_data.append(label)
    
    return feature_data, label_data


def get_components(img, label, flipLR=False, brightnessAlpha=1, brightnessBeta=0, shiftLR=0, shiftUD=0):
    
    # augment data

    if (flipLR):
        img = np.fliplr(img)

    # shift brightness
    img = img.astype('float32')
    img *= brightnessAlpha
    img += brightnessBeta
    img = np.clip(img, 0, 255)

    img = img.astype('uint8')

    # shift eyes lr and ud

    img = cv2.copyMakeBorder(
        img,
        top=abs(shiftUD),
        bottom=abs(shiftUD),
        left=abs(shiftLR),
        right=abs(shiftLR),
        borderType=cv2.BORDER_REPLICATE)

    # crop to original size
    if shiftLR >= 0:
        img = img[:,2*abs(shiftLR):]
    else:
        img = img[:,:-2*abs(shiftLR)]

    if shiftUD > 0:
        img = img[:-2*abs(shiftUD),:]
    else:
        img = img[2*abs(shiftUD):,:]

    # end augment data

    # rescale to 0-1
    img = img / 255.0

    return img, label


def get_augmented_train(features, labels):

    feature_data = []
    label_data = []

    brightnessAlphas = [1]
    brightnessBetas = [0]
    flipLRs = [True, False]
    shiftLRs = [0, 2, -2]
    shiftUDs = [0, 2, -2]

    num_train_pics = len(features) * len(brightnessAlphas) * len(brightnessBetas) * len(flipLRs) * len(shiftLRs) * len(shiftUDs)
    counter = 0

    for i in range (len(features)):
        for flipLR in flipLRs:
            for shiftLR in shiftLRs:
                for shiftUD in shiftUDs:
                    for brightnessAlpha in brightnessAlphas:
                        for brightnessBeta in brightnessBetas:
                            feature, label = get_components(features[i], labels[i], flipLR, brightnessAlpha, brightnessBeta, shiftLR, shiftUD)
                            
                            feature_data.append(feature)
                            label_data.append(label)

                            counter += 1
                            if counter % 100000 == 0:
                                print(str(int((counter / num_train_pics) * 100)) + "% done")

    return feature_data, label_data


def get_augmented_test(features, labels):

    feature_data = []
    label_data = []

    flipLRs = [True, False]

    for i in range (len(features)):
        for flipLR in flipLRs:
            feature, label = get_components(features[i], labels[i], flipLR)
            
            feature_data.append(feature)
            label_data.append(label)

    return feature_data, label_data


def main():

    # get folder to go through
    filtered_file = input("type name of filtered folder: ")
    filtered_file_path = config['path'] + "data/data/filtered/" + filtered_file
    # check if input path is valid
    if not os.path.exists(filtered_file_path):
        print("invalid path")
        exit()

    # get ouptut file path
    file = input("type name of folder to put images: ")
    processed_file_path = config['path'] + "data/data/final/" + file + "/"
    # make output if not exists
    if not os.path.exists(processed_file_path):
        os.makedirs(processed_file_path)

    # train or test
    trainOrTest = input("type train or test: ")
    if trainOrTest != "train" and trainOrTest != "test":
        print("invalid input must be train or test")
        exit()

    # process - check for good data and map orientation to images
    base_feature_data, base_label_data = process(filtered_file_path)
    print("processing finished")

    # augment images - expand base data
    if trainOrTest == "train":
        feature_data, label_data = get_augmented_train(base_feature_data, base_label_data)
    else:
        feature_data, label_data = get_augmented_test(base_feature_data, base_label_data)
    print("augmentation finished")

    # save images - save to compressed file
    np.savez_compressed(processed_file_path + "/" + trainOrTest + "_" + filtered_file, a=feature_data, b=label_data)
    print("saving finished")


if __name__ == '__main__':
    main()