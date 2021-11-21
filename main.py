from application import run
from ray import tune
from functools import partial
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import numpy as np

# model load할 때 필요함
from model import VanillaRNN, LSTM, GRU, RETAIN

if __name__ == '__main__':

    USE_COLAB = False

    PREFIX = ''

    if USE_COLAB:
        from google.colab import drive

        drive.mount('/content/drive')

        PREFIX = '/content/drive/My Drive/'

    x = np.load(PREFIX + 'data/x.npy')
    y = np.load(PREFIX + 'data/y.npy')

    args = {
        'batch_size': tune.choice([128, 256, 512, 1024]),
        'lr': tune.loguniform(1e-5, 1e-2),
        'num_epochs': 4000,
        'weight_decay': 0,
        'dropout': tune.choice([0.2, 0.3, 0.4, 0.5]),
        'hidden_size': 16,
        'num_layers': 2,
        'emb_size': 16,
        'emb_dropout': 0.5,
        'visit_hidden_size': 16,
        'visit_num_layer': 2,
        'var_hidden_size': 16,
        'var_num_layer': 2,
        'n_jitter': 1,
        'jitter_alpha': 0.00125,
        'model': tune.choice(['GRU', 'LSTM', 'RNN', 'RETAIN']),
        'x': x,
        'y': y,
    }
    max_num_epochs = 4000
    num_samples = 16

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        partial(run, DEBUG=False),
        config=args,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))
