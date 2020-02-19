import glob
import os
import pandas as pd
from sklearn.model_selection import KFold


def load_dataset(dataset_path, n_folds, rand_state):
    """
    Parameters
    --------------------
    :param dataset_path:
    :param n_folds:
    :param rand_state:
    :return: list
        List contains split datasets for K-Fold cross-validation
    """
    if os.path.exists(dataset_path + '/data.csv'):
        data = pd.read_csv(dataset_path + '/data.csv')
        all_path = (dataset_path + '/' + data['imageBucket'].astype(str)).to_numpy()
        all_label = data.drop(['imageBucket'], axis=1).to_numpy()

    else:
        # load datapath from path
        all_path = glob.glob(dataset_path + '/**/*')
        all_label = [int(all_path[ibag].replace('\\', '/').split('/')[-2]) for ibag in range(len(all_path))]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=rand_state)
    datasets = []
    labels = []
    for train_idx, test_idx in kf.split(all_path):
        dataset = {}
        label = {}
        dataset['train'] = [all_path[ibag] for ibag in train_idx]
        label['train'] = [all_label[ibag] for ibag in train_idx]
        dataset['test'] = [all_path[ibag] for ibag in test_idx]
        label['test'] = [all_label[ibag] for ibag in test_idx]
        datasets.append(dataset)
        labels.append(label)
    return datasets, labels
