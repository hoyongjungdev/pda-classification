from hyperparameter_search import hyperparameter_search

import numpy as np

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from itertools import product
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


def run(config, DEBUG):
    np.random.seed(0)

    x = config['x']
    y = config['y']

    # dimension
    dim = x.shape[2]

    tv_x, test_x, tv_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)
    #train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y, test_size=0.33, random_state=0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device,'is ready')

    n_splits = 5

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    ba_sum = 0

    for train_idx, val_idx in kf.split(tv_x):
        train_x, val_x = tv_x[train_idx], tv_x[val_idx]
        train_y, val_y = tv_y[train_idx], tv_y[val_idx]

        best_accuracy = hyperparameter_search(config, train_x, train_y, val_x, val_y, device, DEBUG)
        ba_sum += best_accuracy

    test_score = hyperparameter_search(config, tv_x, tv_y, test_x, test_y, device, DEBUG)

    with open('result/result.csv', 'a') as f:
        f.write('{},{},{},{},{}'.format(config['n_jitter'], config['jitter_alpha'], config['model'], ba_sum/n_splits, test_score))
        f.write('\n')
