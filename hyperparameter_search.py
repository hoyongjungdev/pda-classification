from model import VanillaRNN, LSTM, GRU, train_model
from augmentation import augment_jitter
from score import f1

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix

def hyperparameter_search(args, train_x, train_y, validation_x, validation_y, device):
    def create_model(args, feature_size, device):
        if args['model'] == 'GRU':
            return GRU(feature_size, args['hidden_size'], args['num_layers'], args['dropout'], device)
        elif args['model'] == 'LSTM':
            return LSTM(feature_size, args['hidden_size'], args['num_layers'], args['dropout'], device)
        elif args['model'] == 'RNN':
            return VanillaRNN(feature_size, args['hidden_size'], args['num_layers'], args['dropout'], device)

        return None
    
    augmented_x, augmented_y = augment_jitter(args['n_jitter'], args['jitter_alpha'], train_x, train_y)
    augmented_y = np.reshape(augmented_y, (-1, 1))

    validation_x_tensor = torch.Tensor(validation_x).to(device)

    pos_weight =  (train_x == 0).sum() / (train_x == 1).sum()

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    dim = validation_x_tensor.shape[2]

    n_ensemble = 5

    models = []
    y_pred = []

    for i in range(n_ensemble):
        model = create_model(args, dim, device)
        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        model, accuracy = train_model(model, optimizer, criterion, args, augmented_x, augmented_y,
            validation_x, validation_y, device, False, 'result/model.pt')
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

    feature_score = f1(cf)

    print('best accuracy: {}'.format(feature_score))

    return feature_score
