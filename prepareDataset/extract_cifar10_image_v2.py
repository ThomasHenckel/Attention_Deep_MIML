import argparse
import mxnet as mx
import numpy as np
import pickle
import cv2
import os
import random
import pandas as pd

batches_in_cifar10 = 5

def parse_args():
    parser = argparse.ArgumentParser(description='Extract Images from CIFAR-10 into folders, to test out MIML')
    parser.add_argument('--data_path', dest='data_path',
                        help='Data path for CIFAR-10 images',
                        default='../data/cifar-10-batches-py', type=str)
    parser.add_argument('--data_destination', dest='data_destination',
                        help='Destination folder for extracted images',
                        default='../data/cifar-10-miml', type=str)
    parser.add_argument('--images_to_add', dest='images_to_add',
                        help='Number of random images to add to each sample',
                        default=3, type=int)

    args = parser.parse_args()
    return args


def extractImagesAndLabels(path, file):
    f = open(path + file, 'rb')
    dict = pickle.load(f, encoding='latin1')
    images = dict['data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict['labels']
    imagearray = mx.nd.array(images)
    labelarray = mx.nd.array(labels)
    return imagearray, labelarray


def saveCifarImage(array, path, file):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.asnumpy().transpose(1, 2, 0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # make sure path directory exists
    if not os.path.exists(path):
        os.makedirs(path)
    # save to PNG file

    return cv2.imwrite(path + file + ".png", array)


def extractPrincipalImages(dataPath, dataDest):

    # Create Dataframe for bag path and bag labels
    columns = []
    for i in range(10):
        columns.append('label_' + str(i))
    df = pd.DataFrame(columns=columns)

    # loop over each cifar batch in rando order, and add 3-4 images to each bag
    folder = 0
    for batch in range(1, batches_in_cifar10):
        imgarray, lblarray = extractImagesAndLabels(dataPath, "/data_batch_" + str(batch))
        print(imgarray.shape)
        print(lblarray)

        random_order = [i for i in range(len(lblarray))]
        random.shuffle(random_order)

        imaged_added = 0
        images_to_add = 4
        for i in random_order:
            if images_to_add <= imaged_added:
                imaged_added = 0
                folder += 1
                images_to_add = random.randint(3, 4)

            category = lblarray[i].asnumpy()
            category = int(category[0])
            saveCifarImage(imgarray[i], dataDest + '/' + str(folder) + '/', str(category) + "_image" + str(i))
            df.at[folder, 'label_' + str(category)] = 1
            imaged_added += 1

    # Bags path is counted up from 0, so we can just take the index as the bag path
    df['imageBucket'] = df.index

    df = df.fillna(0)
    df.to_csv(dataDest + '/data.csv', index=False)


if __name__ == "__main__":
    args = parse_args()

    print('Called with args:')
    print(args)

    extractPrincipalImages(args.data_path, args.data_destination)
