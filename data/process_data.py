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
    
    img_i = 0
    ori_i = 0
    min_time = 1000000

    # map image data to dictionary
    image_data = os.listdir(filtered_file_path + "/captures/")

    # map orientation data to dictionary
    orientation_data = []
    orientation_file = open(filtered_file_path + '/orientation_data.txt', 'r')
    for line in orientation_file.readlines():
        orientation_data.append(line.strip())
    orientation_file.close()

    while img_i < len(image_data) - 1 and ori_i < len(orientation_data):
        temp_time = int(image_data[img_i].split(".")[0]) - int(orientation_data[ori_i].split("_")[0])

        closer = abs(temp_time) < min_time
        ori_is_before = temp_time > 0
        close_enough = abs(temp_time) <= 250
        
        if close_enough:
            if ori_is_before:
                min_time = abs(temp_time)
                ori_i += 1
                continue
            else:
                if not closer:
                    ori_i -= 1
        else:
            if ori_is_before:
                ori_i += 1
                continue
            else:
                ori_i -= 1
                img_i += 1
                min_time = 1000000
                continue

        img = cv2.imread(filtered_file_path + "/captures/" + image_data[img_i])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # crop out eyes
        # img = img[:,:-25]
        left, right = get_eyes(img)
        eyes = np.concatenate((left, right), axis=1)

        eyes = cv2.resize(eyes, (0,0), fx=0.2, fy=0.2)
        
        # cv2.imshow('eyes', eyes)
        # cv2.waitKey(0)

        # get label data
        label = os.path.basename(orientation_data[ori_i]).replace('.jpg', '').split("_")
        label = np.array([float(label[-2]), float(label[-1])])

        feature_data.append(eyes)
        label_data.append(label)

        min_time = 1000000
        img_i += 1
        ori_i -= 1
    
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

    if (flipLR):
        label = np.array([label[0] * -1, label[1]])

    # scale label to 0-1
    label = (label + 45) / 90.00001
    label = np.clip(label, 0, .999)

    return img, label


def get_augmented_train(features, labels):

    feature_data = []
    label_data = []

    brightnessAlphas = [.8, 1, 1.2]
    brightnessBetas = [0]
    flipLRs = [True, False]
    shiftLRs = [-2, 0, 2]
    shiftUDs = [-2, 0, 2]

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
    if not os.path.exists(filtered_file_path + "/captures/"):
        print("invalid path")
        exit()

    # get ouptut file path
    file = input("type name of folder to put images: ")
    processed_file_path = config['path'] + "data/data/" + file + "/"
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