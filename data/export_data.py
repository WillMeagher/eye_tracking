import os
import numpy as np
import cv2
import random

import sys
sys.path.insert(0, '.')
from config import *

data_path = config['path'] + "data/processed/"
export_path = config['path'] + "data/final"

def get_components(picture, flipLR=False, brightnessAlpha=1, brightnessBeta=0, shiftLR=0, shiftUD=0):
    
    # get img data
    img = cv2.imread(data_path + picture, cv2.IMREAD_GRAYSCALE)

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
    left = img[:,:int(img.shape[1]/2)]
    right = img[:,int(img.shape[1]/2):]

    left = cv2.copyMakeBorder(
        left,
        top=abs(shiftUD),
        bottom=abs(shiftUD),
        left=abs(shiftLR),
        right=abs(shiftLR),
        borderType=cv2.BORDER_REPLICATE)

    right = cv2.copyMakeBorder(
        right,
        top=abs(shiftUD),
        bottom=abs(shiftUD),
        left=abs(shiftLR),
        right=abs(shiftLR),
        borderType=cv2.BORDER_REPLICATE)

    # crop to original size
    if shiftLR >= 0:
        left = left[:,2*abs(shiftLR):]
        right = right[:,2*abs(shiftLR):]
    else:
        left = left[:,:-2*abs(shiftLR)]
        right = right[:,:-2*abs(shiftLR)]

    if shiftUD > 0:
        left = left[:-2*abs(shiftUD),:]
        right = right[:-2*abs(shiftUD),:]
    else:
        left = left[2*abs(shiftUD):,:]
        right = right[2*abs(shiftUD):,:]

    img = np.concatenate((left, right), axis=1)

    # end augment data

    # rescale to 0-1
    img = img / 255.0

    # get label data
    label = picture.replace('.jpg', '').split("_")
    label = np.array([float(label[-2]), float(label[-1])])

    if (flipLR):
        label[0] *= -1

    # scale label to 0-1
    label = (label + 45) / 90.00001
    label = np.clip(label, 0, .999)

    return img, label

def main():

    percent_train = input("percentage of train data: ")

    percent_train = float(percent_train)

    feature_data_train = []
    label_data_train = []

    pictures = os.listdir(data_path)
    random.shuffle(pictures)

    train_pictures = pictures[:int(percent_train * len(pictures))]

    brightnessAlphas = [.8, 1, 1.2]
    brightnessBetas = [0]
    flipLRs = [True, False]
    shiftLRs = [-2, 0, 2]
    shiftUDs = [-2, 0, 2]

    num_train_pics = len(train_pictures) * len(brightnessAlphas) * len(brightnessBetas) * len(flipLRs) * len(shiftLRs) * len(shiftUDs)
    counter = 0

    for picture in train_pictures:
        for flipLR in flipLRs:
            for shiftLR in shiftLRs:
                for shiftUD in shiftUDs:
                    for brightnessAlpha in brightnessAlphas:
                        for brightnessBeta in brightnessBetas:
                            feature, label = get_components(picture, flipLR, brightnessAlpha, brightnessBeta, shiftLR, shiftUD)
                            
                            feature_data_train.append(feature)
                            label_data_train.append(label)

                            counter += 1
                            if counter % 100000 == 0:
                                print(str(int((counter / num_train_pics) * 100)) + "% done")

    print("processing training finished")

    np.savez_compressed(export_path + "/train", a=feature_data_train, b=label_data_train)
    print("saving training finished")


    test_pictures = pictures[int(percent_train * len(pictures)):]

    feature_data_test = []
    label_data_test = []

    for picture in test_pictures:
        for flipLR in flipLRs:
            feature, label = get_components(picture, flipLR)
            
            feature_data_test.append(feature)
            label_data_test.append(label)

    print("processing testing finished")

    np.savez_compressed(export_path + "/test", a=feature_data_test, b=label_data_test)
    print("saving testing finished")


    print("saved " + str(len(feature_data_train)) + " training samples")
    print("saved " + str(len(feature_data_test)) + " testing samples")

if __name__ == '__main__':
    main()