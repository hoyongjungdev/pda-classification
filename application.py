from hyperparameter_search import hyperparameter_search

import numpy as np

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from itertools import product

def run(args_list, DEBUG):
    np.random.seed(0)

    USE_COLAB = False

    PREFIX = ''

    if USE_COLAB:
        from google.colab import drive
        drive.mount('/content/drive')

        PREFIX = '/content/drive/My Drive/'

    x = np.load(PREFIX + 'data/x.npy')
    y = np.load(PREFIX + 'data/y.npy')

    # dimension
    dim = x.shape[2]

    tv_x, test_x, tv_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)
    #train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y, test_size=0.33, random_state=0)

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

        n_splits = 5

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

        ba_sum = 0

        for train_idx, val_idx in kf.split(tv_x):
            train_x, val_x = tv_x[train_idx], tv_x[val_idx]
            train_y, val_y = tv_y[train_idx], tv_y[val_idx]

            best_accuracy = hyperparameter_search(args, train_x, train_y, val_x, val_y, device, DEBUG)
            ba_sum += best_accuracy

        with open(PREFIX+'result/result.csv','a') as f:
            f.write('{},{},{},{}'.format(args['n_jitter'], args['jitter_alpha'], args['model'], ba_sum/n_splits))
            f.write('\n')
