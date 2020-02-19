import pandas as pd


def load_dataset(dataset_path):
    """
    Parameters
    --------------------
    :param dataset_path:
    :return: list, List
        List contains paths to the bags
        List containing the labels for each bag
    """
    data = pd.read_csv(dataset_path + '/data.csv')
    dataset = (dataset_path + '/' + data['imageBucket'].astype(str)).to_numpy()
    labels = data.drop(['imageBucket'], axis=1).to_numpy()

    return dataset, labels
