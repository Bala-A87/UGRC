import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter('ignore', category=ConvergenceWarning)
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
import os
from pathlib import Path
from scripts.utils import make_dataloader, EarlyStopping, plot_train_history
import json
from scripts.metrics import NegMeanSquaredError
from scripts.train import train_model
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scripts.test import predict
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_squared_error, make_scorer
from scripts.ntk import get_ntk_feature_matrix

"""
DATA AND CONSTANTS
"""

parser = argparse.ArgumentParser()
parser.add_argument('-deg', type=int, default=4, help='Degree of polynomial to generate')
parser.add_argument('-svm', action='store_true', help='Whether to run with svm also')

args = parser.parse_args()

log_path = Path(f'logs/direct-func-poly/degree_{args.deg}.txt')
if log_path.is_file():
    os.remove(log_path)
if not Path('logs/direct-func-poly/').is_dir():
    if not Path('logs/').is_dir():
        os.mkdir('logs/')
    os.mkdir('logs/direct-func-poly/')

if not Path('plots/direct-func-poly/').is_dir():
    if not Path('plots/').is_dir():
        os.mkdir('plots/')
    os.mkdir('plots/direct-func-poly/')
    os.mkdir('plots/direct-func-poly/train-val-loss-score/')
    os.mkdir('plots/direct-func-poly/nn-fit/')
    os.mkdir('plots/direct-func-poly/svm-fit/')

plot_dir = 'plots/direct-func-poly/'
plot_file_name = f'degree_{args.deg}'
evolution_plot_path = plot_dir + 'train-val-loss-score/' + plot_file_name + '.png'
nn_fit_plot_path = plot_dir + 'nn-fit/' + plot_file_name + '.png'
svm_fit_plot_path = plot_dir + 'svm-fit/' + plot_file_name + '.png'

def add_log(log_str: str, file_name: Path = log_path) -> None:
    with open(file_name, 'a') as f:
        f.write(log_str+'\n')

DATA_SIZE = 10000
RANGE = 2
Y_LIM = (-1, 1)

class SimpleNN(nn.Module):
    def __init__(self, width) -> None:
        super().__init__()
        self.width = width
        self.input = nn.Sequential(
            nn.Linear(in_features=1, out_features=width),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(in_features=width, out_features=1)
        )

    def forward(self, x):
        return self.output(self.input(x))
    
    def clone(self):
        new_model = SimpleNN(self.width)
        new_model.load_state_dict(self.state_dict())
        return new_model

torch.manual_seed(844)
X = torch.randn((DATA_SIZE, 1))
add_log(f'X.shape == {X.shape}')

scale_factor = RANGE / X.abs().max()
add_log(f'scale_factor == {scale_factor}')
X *= scale_factor
roots = torch.linspace(-RANGE, RANGE, args.deg)
Y = torch.ones_like(X)
for root in roots:
    Y *= (X - root)
add_log(f'Y.shape == {Y.shape}')

X_training, X_test, Y_training, Y_test = train_test_split(X, Y, test_size=0.1, random_state=13)
X_train, X_val, Y_train, Y_val = train_test_split(X_training, Y_training, test_size=1/9, random_state=9173)
add_log(f'X_train.shape == {X_train.shape}, Y_train.shape == {Y_train.shape}')
add_log(f'X_val.shape == {X_val.shape}, Y_val.shape == {Y_val.shape}')
add_log(f'X_test.shape == {X_test.shape}, Y_test.shape == {Y_test.shape}')

# Y_LIM = (Y_train.min(), Y_train.max())
# plt.scatter(X_train.sum(dim=1), Y_train)
# plt.xlim(-RANGE, RANGE)
# plt.ylim(Y_LIM)
# plt.show()

# quit(0)

train_dataloader, val_dataloader = make_dataloader(X_train, Y_train, 32, True), make_dataloader(X_val, Y_val, 32, True)

"""
NN CROSS-VAL AND TRAINING
"""

device = 'cpu'
widths = [128, 256, 512, 1024]
etas = [1e-3, 1e-2, 1e-1, 1.]
weight_decays = np.logspace(-4, 2, 7).tolist() + [0.0]

config_file = Path(f'configs/nn/direct-func-poly/degree_{args.deg}.json')
if config_file.is_file():
    with open(config_file, 'r') as f:
        best_config = json.load(f)
    best_width = best_config['width']
    best_eta = best_config['eta']
    best_weight_decay = best_config['weight_decay']
    best_score = best_config['score']
else:
    if not Path('configs/nn/direct-func-poly/').is_dir():
        if not Path('configs/nn/').is_dir():
            if not Path('configs/').is_dir():
                os.mkdir('configs/')
            os.mkdir('configs/nn/')
        os.mkdir('configs/nn/direct-func-poly/')
    best_score = -torch.inf
    best_width = None
    best_eta = None
    best_weight_decay = None

total_count = len(widths) * len(etas) * len(weight_decays)
curr_count = 0
EPOCHS = 20

if best_width is None:
    add_log(f'Cross-validating across {total_count} models.')

    for width in widths:
        for eta in etas:
            for weight_decay in weight_decays:
                curr_count += 1
                torch.random.manual_seed(47647)
                model = SimpleNN(width).to(device)
                optimizer = torch.optim.Adadelta(params=model.parameters(), lr=eta, weight_decay=weight_decay)
                history = train_model(
                    model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    loss_fn=torch.nn.L1Loss(),
                    optimizer=optimizer,
                    metric=NegMeanSquaredError(),
                    epochs=EPOCHS,
                    verbose=0,
                    device=device
                )
                curr_score = history['val_score'][-1]
                if curr_score > best_score:
                    best_score = curr_score
                    best_width = width
                    best_eta = eta
                    best_weight_decay = weight_decay
                add_log(f'[{curr_count}/{total_count}]\tWidth:{width}, lr:{eta}, w_d:{weight_decay} => Score:{curr_score:.6f}')
    best_config = {
        'score': float(best_score),
        'width': best_width,
        'eta': best_eta,
        'weight_decay': best_weight_decay,
    }
    with open(config_file, 'w') as f:
        json.dump(best_config, f)

add_log(f'\nBest validation score after {EPOCHS} epochs: {best_score:.6f}. Best configuration:')
add_log(f'Width:{best_width}, lr:{best_eta}, w_d:{best_weight_decay}')

torch.random.manual_seed(47647)
best_model_nn = SimpleNN(best_width).to(device)

loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.Adadelta(params=best_model_nn.parameters(), lr=best_eta, weight_decay=best_weight_decay)
metric = NegMeanSquaredError()
early_stop = EarlyStopping(patience=50, min_delta=1e-4)

history = train_model(
    model=best_model_nn,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    metric=metric,
    epochs=500,
    early_stopping=early_stop,
    device=device
)

plot_train_history(history, evolution_plot_path)

preds_train_nn, preds_val_nn, preds_test_nn = predict(best_model_nn, X_train, device), predict(best_model_nn, X_val, device), predict(best_model_nn, X_test, device)
add_log(f'preds_train_nn.shape == {preds_train_nn.shape}, preds_val_nn.shape == {preds_val_nn.shape}, preds_test_nn.shape == {preds_test_nn.shape}')

score_train, score_val, score_test = metric(preds_train_nn, Y_train), metric(preds_val_nn, Y_val), metric(preds_test_nn, Y_test)
add_log(f'score_train == {score_train}, score_val == {score_val}, score_test == {score_test}')

plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.scatter(X_train, Y_train, c='g', label='True', s=4, alpha=0.2)
plt.scatter(X_train, preds_train_nn, c='r', label='Predicted', s=4, alpha=0.2)
plt.title('Train')
plt.xlim(-RANGE, RANGE)
plt.ylim(Y_LIM)
plt.legend()

plt.subplot(122)
plt.scatter(X_test, Y_test, c='g', label='True', s=4, alpha=0.2)
plt.scatter(X_test, preds_test_nn, c='r', label='Predicted', s=4, alpha=0.2)
plt.title('Test')
plt.xlim(-RANGE, RANGE)
plt.ylim(Y_LIM)
plt.legend()

plt.suptitle('NN')
plt.savefig(nn_fit_plot_path)

"""
SVM CROSS-VAL AND TRAINING
"""
if args.svm:

    X_train_ntk = get_ntk_feature_matrix(X_train, best_model_nn)
    X_val_ntk = get_ntk_feature_matrix(X_val, best_model_nn)

    model_base_ntk = LinearSVR()
    params_ntk = {
        'C': np.logspace(-4, 2, 7)
    }

    gammas = np.logspace(-4, 4, 9).tolist() + ['scale', 'auto']
    model_base_rbf = SVR(kernel='rbf', max_iter=int(1e4))
    params_rbf = {
        'C': np.logspace(-4, 2, 7),
        'gamma': gammas
    }

    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    model_cv_ntk = GridSearchCV(
        estimator=model_base_ntk,
        param_grid=params_ntk,
        scoring=scorer,
        n_jobs=1,
        cv=4,
        refit=False,
        verbose=3
    )
    model_cv_ntk.fit(X_train_ntk, Y_train.squeeze())
    best_params_ntk = model_cv_ntk.best_params_
    best_score_ntk = max(model_cv_ntk.cv_results_['mean_test_score'])
    add_log(f'Best params for NTK: {best_params_ntk}')
    add_log(f'Best score for NTK: {best_score_ntk}')

    model_cv_rbf = GridSearchCV(
        estimator=model_base_rbf,
        param_grid=params_rbf,
        scoring=scorer,
        n_jobs=1,
        cv=4,
        refit=False,
        verbose=3
    )
    model_cv_rbf.fit(X_train, Y_train.squeeze())
    best_params_rbf = model_cv_rbf.best_params_
    best_score_rbf = max(model_cv_rbf.cv_results_['mean_test_score'])
    add_log(f'Best params for RBF: {best_params_rbf}')
    add_log(f'Best score for RBF: {best_score_rbf}')

    best_kernel = 'ntk' if best_score_ntk >= best_score_rbf else 'rbf'
    add_log(f'Best kernel: {best_kernel}')

    if best_kernel == 'ntk':
        X_test_ntk = get_ntk_feature_matrix(X_test, best_model_nn)
        best_model_km = LinearSVR(C=best_params_ntk['C'])
        best_model_km.fit(X_train_ntk, Y_train.squeeze())
        preds_train_km, preds_val_km, preds_test_km = best_model_km.predict(X_train_ntk), best_model_km.predict(X_val_ntk), best_model_km.predict(X_test_ntk)
    else:
        best_model_km = SVR(C=best_params_rbf['C'], gamma=best_params_rbf['gamma'])
        best_model_km.fit(X_train, Y_train.squeeze())
        preds_train_km, preds_val_km, preds_test_km = best_model_km.predict(X_train), best_model_km.predict(X_val), best_model_km.predict(X_test)

    add_log(f'preds_train_km.shape == {preds_train_km.shape}, preds_val_km.shape == {preds_val_km.shape}, preds_test_km.shape == {preds_test_km.shape}')

    score_train, score_val, score_test = mean_squared_error(Y_train.squeeze(), preds_train_km), mean_squared_error(Y_val.squeeze(), preds_val_km), mean_squared_error(Y_test.squeeze(), preds_test_km)
    add_log(f'score_train == {score_train}, score_val == {score_val}, score_test == {score_test}')

    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.scatter(X_train, Y_train, c='g', label='True', s=4, alpha=0.2)
    plt.scatter(X_train, preds_train_km, c='r', label='Predicted', s=4, alpha=0.2)
    plt.title('Train')
    plt.xlim(-RANGE, RANGE)
    plt.ylim(Y_LIM)
    plt.legend()

    plt.subplot(122)
    plt.scatter(X_test, Y_test, c='g', label='True', s=4, alpha=0.2)
    plt.scatter(X_test, preds_test_km, c='r', label='Predicted', s=4, alpha=0.2)
    plt.title('Test')
    plt.xlim(-RANGE, RANGE)
    plt.ylim(Y_LIM)
    plt.legend()

    plt.suptitle(f'SVM ({best_kernel})')
    plt.savefig(svm_fit_plot_path)
