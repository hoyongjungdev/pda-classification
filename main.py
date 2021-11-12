from application import run

# model load할 때 필요함
from model import VanillaRNN, LSTM, GRU

if __name__ == '__main__':
    args_list = {
        'batch_size': [1024],
        'lr': [1e-4],
        'num_epochs': [4000],
        'weight_decay': [0],
        'dropout': [0.5],
        'hidden_size': [16],
        'num_layers': [2],
        'n_jitter': [1],
        'jitter_alpha': [0.00125],
        'model': ['RNN', 'LSTM', 'GRU'],
    }

    run(args_list, False)
