import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


def process_data():

    # load data
    x_train = np.load('kmnist-train-imgs.npz')['arr_0']
    y_train = np.load('kmnist-train-labels.npz')['arr_0']
    x_test = np.load('kmnist-test-imgs.npz')['arr_0']
    y_test = np.load('kmnist-test-labels.npz')['arr_0']

    # normalize pixel values between 0 and 1
    x_train = x_train / 255
    x_test = x_test / 255

    # one hot encode labels
    ohc = OneHotEncoder(sparse=False)
    y_train = ohc.fit_transform(y_train.reshape(-1, 1)).astype(int)
    y_test = ohc.fit_transform(y_test.reshape(-1, 1)).astype(int)
    # print(np.unique(y_train, return_counts=True))

    # print(x_train.shape, y_train.shape)
    # print(x_test.shape, y_test.shape)
    # print()

    # split for train_a, test_a
    x_train_a, x_test_a, y_train_a, y_test_a = train_test_split(
        x_train, y_train, test_size=.33, stratify=y_train)

    # print(x_train_a.shape, y_train_a.shape)
    # print(x_test_a.shape, y_test_a.shape)
    # print()

    # split for train_b, test_b
    x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(
        x_train_a, y_train_a, test_size=.33, stratify=y_train_a)

    # print(x_train_b.shape, y_train_b.shape)
    # print(x_test_b.shape, y_test_b.shape)
    # print()

    return {
        'full': [x_train, x_test, y_train, y_test],
        'split_a': [x_train_a, x_test_a, y_train_a, y_test_a],
        'split_b': [x_train_b, x_test_b, y_train_b, y_test_b],
    }


# def cross_validation_indecies(n, x_train, y_train):
#     skf = StratifiedKFold(n_splits=n)
#     ind = skf.split(x_train, y_train)
#     return ind

# dataDict = process_data()

# anInd = cross_validation_indecies(3, np.zeros(
#     dataDict['split_b'][0].shape[0]), dataDict['split_b'][2])
# for i, j in anInd:
#     print(i, j)


