from score import f1

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import confusion_matrix

def to_dataloader(args, x, y, device):
    x_tensor = torch.Tensor(x).to(device) # transform to torch tensor
    y_tensor = torch.Tensor(y).to(device)

    dataset = TensorDataset(x_tensor, y_tensor) # create your datset
    return DataLoader(dataset, batch_size=args['batch_size']) # create your dataloader

# basic Vanilla RNN
class VanillaRNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, dropout, device):
    super(VanillaRNN, self).__init__()
    self.device = device
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.d = 1
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False).to(self.device)
    self.bn = nn.BatchNorm1d(self.d * hidden_size).to(self.device)
    self.fc = nn.Sequential(nn.Linear(self.d * hidden_size, 1)).to(self.device)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers*self.d, x.size()[0], self.hidden_size).to(self.device) # 초기 hidden state
    
    out, _ = self.rnn(x, h0)
    out = out[:,-1]

    # normalization + layering
    out = self.fc(
        self.bn(out)
      )
    return out

# LSTM
class LSTM(nn.Module): # LSTM with 1 seq
  def __init__(self, input_size, hidden_size, num_layers, dropout, device):
    super(LSTM, self).__init__()
    self.device = device
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.d = 1

    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False).to(self.device)
    self.bn = nn.BatchNorm1d(self.d * hidden_size).to(self.device)
    self.fc = nn.Sequential(nn.Linear(self.d * hidden_size, 1)).to(self.device)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers*self.d, x.size()[0], self.hidden_size).to(self.device) # 초기 hidden state
    c0 = torch.zeros(self.num_layers*self.d, x.size()[0], self.hidden_size).to(self.device) # 초기 cell state
    
    out, _ = self.lstm(x, (h0, c0))
    out = out[:,-1]

    # normalization + layering
    out = self.fc(
        self.bn(out)
      )
    return out

# GRU
class GRU(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, dropout, device):
    super(GRU, self).__init__()
    self.input_size = input_size
    self.device = device
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.d = 1

    self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False).to(self.device)
    self.bn = nn.BatchNorm1d(self.d * hidden_size).to(self.device)
    self.fc = nn.Sequential(nn.Linear(self.d * hidden_size, 1)).to(self.device)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers*self.d, x.size()[0], self.hidden_size).to(self.device) # 초기 hidden state
    
    out, _ = self.gru(x, h0)
    out = out[:,-1]

    # normalization + layering
    out = self.fc(
        self.bn(out)
      )
    return out

def train_model(model, optimizer, criterion, args, train_x, train_y, validation_x, validation_y, device, DEBUG, path):
    train_dataloader = to_dataloader(args, train_x, train_y, device)
    validation_dataloader = to_dataloader(args, validation_x, validation_y, device)
    running_loss = []

    val_acc = []
    train_acc = []

    best_acc = -1

    for epoch in range(args['num_epochs']):
        if DEBUG:
            if (epoch % 1000 == 0 and epoch!=0) or epoch==args['num_epochs']-1:
                print('epoch: {}'.format(epoch))
                plt.plot(train_acc)
                plt.plot(val_acc)
                plt.legend(['train', 'validation'])
                plt.show()
    
        # train mode
        model.train()

        for seq, target in train_dataloader:
            out = model(seq)
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
        
        # eval mode
        model.eval()

        if epoch % 10 == 0 and epoch > 0:
            val_cf = np.zeros((2, 2))
            train_cf = np.zeros((2, 2))

            with torch.no_grad():
                for seq, target in validation_dataloader:
                    out = model(seq)
                    result = (out > 0.5).float()
                    cf = confusion_matrix(target, result)

                    val_cf += cf

                for seq, target in train_dataloader:
                    out = model(seq)
                    result = (out > 0.5).float()
                    cf = confusion_matrix(target, result)

                    train_cf += cf                 

                val_f1 = f1(val_cf)
                train_f1 = f1(train_cf)

                val_acc.append(val_f1)
                train_acc.append(train_f1)

                # saving best model
                if val_f1 > best_acc:
                    best_acc = val_f1

                    torch.save(model, path)

    model = torch.load(path)
    model.eval()

    print("f1:", best_acc)

    return model, best_acc
