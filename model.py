from score import f1

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# RETAIN
class RETAIN(nn.Module):
  def __init__(self, input_size, emb_size, emb_dropout, visit_hidden_size, visit_num_layer, var_hidden_size, var_num_layer, dropout, device):
    super(RETAIN, self).__init__()
    self.device = device
    self.input_size = input_size
    self.visit_hidden_size = visit_hidden_size
    self.visit_num_layer = visit_num_layer
    self.visit_attn_output_size = 1
    self.var_hidden_size = var_hidden_size
    self.var_num_layer = var_num_layer
    self.dropout = dropout

    # data embedding
    self.emb_size = emb_size
    self.embed_layer = nn.Linear(input_size, emb_size).to(self.device)
    self.embed_dropout = nn.Dropout(emb_dropout)
    # visit RNN
    self.visit_rnn = nn.GRU(emb_size, visit_hidden_size, visit_num_layer, batch_first=True).to(self.device)
    self.visit_level_attention = nn.Linear(visit_hidden_size, self.visit_attn_output_size).to(self.device)
    # var RNN
    self.var_rnn = nn.GRU(emb_size, var_hidden_size, var_num_layer, batch_first=True).to(self.device)
    self.variable_level_attention = nn.Linear(var_hidden_size, emb_size).to(self.device)
    # output layer
    self.output_layer = nn.Sequential(nn.Linear(emb_size, 1), nn.Sigmoid()).to(self.device)
    self.out_dropout = nn.Dropout(dropout)

  def forward(self, x):
    # 1. embedding
    emb_x = self.embed_layer(x)
    emb_x = self.embed_dropout(emb_x)
    # 2. visit level attention
    visit_h0 = torch.zeros(self.visit_num_layer, x.size()[0], self.visit_hidden_size).to(self.device)
    visit_rnn_output, _ = self.visit_rnn(torch.flip(emb_x, [0]), visit_h0) # in reverse order
    alpha = self.visit_level_attention(torch.flip(visit_rnn_output, [0])) # α (scalar)
    visit_attn_w = F.softmax(alpha, dim=0)
    # 3. var level attention
    var_h0 = torch.zeros(self.var_num_layer, x.size()[0], self.var_hidden_size).to(self.device)
    var_rnn_output, _ = self.var_rnn(torch.flip(emb_x, [0]), var_h0) # in reverse order
    beta = self.variable_level_attention(torch.flip(var_rnn_output, [0])) # β (vector)
    var_attn_w = torch.tanh(beta)
    # 4. context vector
    attn_w = visit_attn_w * var_attn_w
    c = torch.sum(attn_w * emb_x, dim=1)
    c = self.out_dropout(c)
    # 5. prediction
    output = self.output_layer(c)
    output = F.softmax(output, dim=1)

    return output

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
                    cf = confusion_matrix(target.cpu(), result.cpu())

                    val_cf += cf

                for seq, target in train_dataloader:
                    out = model(seq)
                    result = (out > 0.5).float()
                    cf = confusion_matrix(target.cpu(), result.cpu())

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
