import argparse
import random
import pandas as pd
import glob2
import os
from shutil import copyfile
from sklearn.preprocessing import LabelEncoder

batches_in_cifar10 = 5


def parse_args():
    parser = argparse.ArgumentParser(description='Construct bags of images, to test out MIML')
    parser.add_argument('--data_path', dest='data_path',
                        help='Data path for natural images dataset',
                        default='../data/natural-images', type=str)
    parser.add_argument('--data_destination', dest='data_destination',
                        help='Destination folder for extracted images',
                        default='../data/natural-images-miml', type=str)
    parser.add_argument('--max_images_to_add', dest='max_images_to_add',
                        help='Number of images to add to each bag',
                        default=3, type=int)
    parser.add_argument('--min_images_to_add', dest='min_images_to_add',
                        help='Number of images to add to each bag',
                        default=3, type=int)

    args = parser.parse_args()
    return args


def encodeLabels(propertyToEncode, encoder=None):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(propertyToEncode)
    code = encoder.transform(propertyToEncode)

    return code, encoder


def extractPrincipalImages(dataPath, dataDest):
    imgarray = glob2.glob(dataPath + '/**/*.jpg')

    lblarray = [n.replace('\\', '/').split('/')[-2] for n in imgarray]
    lblarray = encodeLabels(lblarray)[0]

    # Create Dataframe for bag path and bag labels
    columns = []
    for i in range(len(set(lblarrayb))):
        columns.append('label_' + str(i))
    df = pd.DataFrame(columns=columns)

    random_order = [i for i in range(len(imgarray))]
    random.shuffle(random_order)

    folder = 0

    imaged_added = 0
    images_to_add = args.max_images_to_add
    for i in random_order:
        if images_to_add <= imaged_added:
            imaged_added = 0
            folder += 1
            images_to_add = random.randint(args.min_images_to_add, args.max_images_to_add)

        # category = lblarray[i].asnumpy()
        category = lblarray[i]

        if not os.path.exists(dataDest + '/' + str(folder)):
            os.makedirs(dataDest + '/' + str(folder))
        copyfile(imgarray[i], dataDest + '/' + str(folder) + '/' + str(category) + "_image_" + str(i) + '.jpg')
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
