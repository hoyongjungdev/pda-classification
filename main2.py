from model import VanillaRNN, LSTM, GRU, RETAIN, train_model2
from augmentation import augment_jitter, augment_scale
from score import f1

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import numpy as np

import random

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix


def hyperparameter_search2(args, train_x, train_y, validation_x, validation_y, device, DEBUG):
    def create_model(args, feature_size, device):
        if args['model'] == 'GRU':
            return GRU(feature_size, args['hidden_size'], args['num_layers'], args['dropout'], device)
        elif args['model'] == 'LSTM':
            return LSTM(feature_size, args['hidden_size'], args['num_layers'], args['dropout'], device)
        elif args['model'] == 'RNN':
            return VanillaRNN(feature_size, args['hidden_size'], args['num_layers'], args['dropout'], device)
        elif args['model'] == 'RETAIN':
            return RETAIN(feature_size, args['emb_size'], args['emb_dropout'], args['visit_hidden_size'],
                          args['visit_num_layer'],
                          args['var_hidden_size'], args['var_num_layer'], args['dropout'], device)
        return None

    augmented_x, augmented_y = augment_scale(args['n_jitter'], args['jitter_alpha'], train_x, train_y)
    augmented_y = np.reshape(augmented_y, (-1, 1))

    validation_x_tensor = torch.Tensor(validation_x).to(device)

    pos_weight = (train_x == 0).sum() / (train_x == 1).sum()

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    dim = validation_x_tensor.shape[2]

    n_ensemble = 1

    models = []
    y_pred = []

    for i in range(n_ensemble):
        model = create_model(args, dim, device)
        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        model, accuracy = train_model2(model, optimizer, criterion, args, augmented_x, augmented_y,
                                      validation_x, validation_y, device, DEBUG, 'result/model.pt')
        models.append(model)

        out = model(validation_x_tensor)
        result = (out > 0.5).float()
        y_pred.append(result)

    result = torch.empty(len(validation_x))

    for i in range(len(validation_x)):
        one_count = 0
        zero_count = 0

        for j in range(len(y_pred)):
            if y_pred[j][i] == 0:
                zero_count += 1
            else:
                one_count += 1

        if zero_count > one_count:
            result[i] = 0.0
        else:
            result[i] = 1.0

    cf = confusion_matrix(validation_y, result)
    print(cf)

    feature_score = f1(cf)

    print('best accuracy: {}'.format(feature_score))

    return feature_score

DEBUG = False

config = {
    'batch_size': 128,
    'lr': 0.0008131,
    'num_epochs': 4000,
    'weight_decay': 0,
    'dropout': 0.3,
    'hidden_size': 16,
    'num_layers': 2,
    'emb_size': 16,
    'emb_dropout': 0.5,
    'visit_hidden_size': 16,
    'visit_num_layer': 2,
    'var_hidden_size': 16,
    'var_num_layer': 2,
    'n_jitter': 5000,
    'jitter_alpha': 0.00125,
    'model': 'LSTM',
}

PREFIX = ''
x = np.load(PREFIX + 'data/x.npy')
y = np.load(PREFIX + 'data/y.npy')

random_seed = 0

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

# dimension
dim = x.shape[2]

tv_x, test_x, tv_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device,'is ready')

test_score = hyperparameter_search2(config, tv_x, tv_y, test_x, test_y, device, DEBUG)

with open('result/result.csv', 'a') as f:
    f.write('{},{},{}'.format(config['n_jitter'], config['jitter_alpha'], test_score))
    f.write('\n')

