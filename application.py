from hyperparameter_search import hyperparameter_search

import numpy as np

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold

from itertools import product

def run(args_list):
    np.random.seed(0)

    USE_COLAB = False

    PREFIX = ''

    DEBUG = False

    if USE_COLAB:
        from google.colab import drive
        drive.mount('/content/drive')

        PREFIX = '/content/drive/My Drive/'

    # finger_data 폴더에서 다운로드

    x = np.load(PREFIX + 'data/x.npy')
    y = np.load(PREFIX + 'data/y.npy')

    # dimension
    dim = x.shape[2]

    tv_x, test_x, tv_y, test_y = train_test_split(x, y, test_size=0.25, random_state=0)
    #train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y, test_size=0.33, random_state=0)

    min_val = np.empty(dim)
    max_val = np.empty(dim)

    for i in range(dim):
        feature = tv_x[:,:,i]
        min_val[i] = feature.min()
        max_val[i] = feature.max()

        if min_val[i] == max_val[i]:
            max_val[i] = 1
            min_val[i] = 0

    #print(min_val)
    #print(max_val - min_val)

    tv_x = (tv_x - min_val) / (max_val - min_val)
    test_x = (test_x - min_val) / (max_val - min_val)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device,'is ready')

    values = list(args_list.values())
    hyperparameters = list(product(*values))

    for hyperparameter in hyperparameters:
        args = dict()

        for i in range(len(args_list)):
            key = list(args_list.keys())[i]
            args[key] = hyperparameter[i]

        print(args)

        n_splits = 3
        n_repeats = 5

        kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)

        ba_sum = 0

        for train_idx, val_idx in kf.split(tv_x):
            train_x, val_x = tv_x[train_idx], tv_x[val_idx]
            train_y, val_y = tv_y[train_idx], tv_y[val_idx]

            best_accuracy = hyperparameter_search(args, train_x, train_y, val_x, val_y, device)
            ba_sum += best_accuracy

        with open(PREFIX+'result/result.csv','a') as f:
            f.write('{},{},{},{}'.format(args['n_jitter'], args['jitter_alpha'], args['model'], ba_sum/n_splits/n_repeats))
            f.write('\n')
