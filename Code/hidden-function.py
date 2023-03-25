import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter('ignore', category=ConvergenceWarning)
import argparse
from torch.distributions import MultivariateNormal
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
parser.add_argument('-cycles', type=int, default=4, help='Number of cycles of the sine function to generate')
parser.add_argument('-corr', action='store_true', help='Whether to draw data from distribution with non-identity covariance')
parser.add_argument('-mode', type=str.lower, choices=['reg', 'stop_first', 'first_last', 'reinit'], default='reg', help='Mode of training to use')

args = parser.parse_args()

log_path = Path(f'logs/hidden-function_{args.cycles}cycles_{args.mode}{"_corr" if args.corr else ""}')
if log_path.is_file():
    os.remove(log_path)

def add_log(log_str: str, file_name: Path = log_path) -> None:
    with open(file_name, 'a') as f:
        f.write(log_str+'\n')

DATA_SIZE = 10000
NUM_FEATURES = 25

class SimpleNN(nn.Module):
    def __init__(self, width, activation) -> None:
        super().__init__()
        self.width = width
        self.activation = activation
        self.input = nn.Sequential(
            nn.Linear(in_features=NUM_FEATURES, out_features=NUM_FEATURES),
            activation()
        )
        self.hidden = nn.Sequential(
            nn.Linear(in_features=NUM_FEATURES, out_features=width),
            activation()
        )
        self.output = nn.Sequential(
            nn.Linear(in_features=width, out_features=1)
        )

    def forward(self, x):
        return self.output(self.hidden(self.input(x)))
    
    def clone(self):
        new_model = SimpleNN(self.width, self.activation)
        new_model.load_state_dict(self.state_dict())
        return new_model

    def freeze_first(self) -> None:
        for param in self.input.parameters():
            param.requires_grad = False
        
    def freeze_last(self) -> None:
        for param in self.hidden.parameters():
            param.requires_grad = False
        for param in self.output.parameters():
            param.requires_grad = False
    
    def revive_last(self) -> None:
        for param in self.hidden.parameters():
            param.requires_grad = True
        for param in self.output.parameters():
            param.requires_grad = True
    
    def reinit_last(self) -> None:
        for child in self.hidden.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()
        for child in self.output.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()


extra_cov = 1.0 if args.corr else 0.0

distribution = MultivariateNormal(torch.zeros(NUM_FEATURES), torch.eye(NUM_FEATURES) + extra_cov)

torch.manual_seed(844) 
X = distribution.sample((DATA_SIZE,))
add_log(f'X.shape == {X.shape}')

scale_factor = args.cycles * torch.pi / X.sum(dim=1).max()
add_log(f'scale_factor == {scale_factor}')

Y = torch.sin(scale_factor * X.sum(dim=1)).reshape(-1,)
add_log(f'Y.shape == {Y.shape}')

X_training, X_test, Y_training, Y_test = train_test_split(X, Y, test_size=1000, random_state=13)
X_train, X_val, Y_train, Y_val = train_test_split(X_training, Y_training, test_size=1000, random_state=9173)
add_log(f'X_train.shape == {X_train.shape}, Y_train.shape == {Y_train.shape}')
add_log(f'X_val.shape == {X_val.shape}, Y_val.shape == {Y_val.shape}')
add_log(f'X_test.shape == {X_test.shape}, Y_test.shape == {Y_test.shape}')

train_dataloader, val_dataloader = make_dataloader(X_train, Y_train, 16, True), make_dataloader(X_val, Y_val, 16, True)

"""
NN CROSS-VAL AND TRAINING
"""

device = 'cpu'

def get_cosine_angles(model: SimpleNN, ref_vector: torch.Tensor = torch.ones(NUM_FEATURES)):
    weights = next(model.input.parameters()).detach()
    return torch.tensor([torch.dot(weight, ref_vector) / (torch.linalg.norm(ref_vector) * torch.linalg.norm(weight)) for weight in weights])

# For NN crossval, just load config based on 4 combos from cycles and corr
# If config absent, have to do crossval
# SVM crossval has to be done anyway

widths = [128, 256, 512, 1024, 2048, 4096]
activations = ['relu', 'tanh']
etas = [1e-4, 1e-3, 1e-2, 1e-1, 1.]
weight_decays = np.logspace(-4, 2, 7).tolist() + [0.0]

config_file = Path(f'configs/nn/hidden-function/{"corr_" if args.corr else ""}{args.cycles}cycles.json')
if config_file.is_file():
    with open(config_file, 'r') as f:
        best_config = json.load(f)
    best_width = best_config['width']
    best_activation = best_config['activation']
    best_eta = best_config['eta']
    best_weight_decay = best_config['weight_decay']
    best_score = best_config['score']
else:
    best_score = -torch.inf
    best_width = None
    best_activation = None
    best_eta = None
    best_weight_decay = None

total_count = len(widths) * len(activations) * len(etas) * len(weight_decays)
curr_count = 0
EPOCHS = 50

if best_width is None:
    add_log(f'Cross-validating across {total_count} models.')

    for width in widths:
        for activation in activations:
            for eta in etas:
                for weight_decay in weight_decays:
                    curr_count += 1
                    torch.random.manual_seed(47647)
                    activation_fn = torch.nn.ReLU if activation == 'relu' else torch.nn.Tanh
                    model = SimpleNN(width, activation_fn).to(device)
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
                        best_activation = activation
                        best_eta = eta
                        best_weight_decay = weight_decay
                    print(f'[{curr_count}/{total_count}]\tWidth:{width}, Actn.:{activation}, lr:{eta}, w_d:{weight_decay} => Score:{curr_score:.6f}')
    best_config = {
        'score': float(best_score),
        'width': best_width,
        'activation': best_activation,
        'eta': best_eta,
        'weight_decay': best_weight_decay,
    }
    with open(config_file, 'w') as f:
        json.dump(best_config, f)

add_log(f'\nBest validation score after {EPOCHS} epochs: {best_score:.6f}. Best configuration:')
add_log(f'Width:{best_width}, Actn.:{best_activation}, lr:{best_eta}, w_d:{best_weight_decay}')

torch.random.manual_seed(47647)
best_activation_fn = torch.nn.ReLU if best_activation == 'relu' else torch.nn.Tanh
best_model_nn = SimpleNN(best_width, best_activation_fn).to(device)
model_0 = best_model_nn.clone()

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
    strategy=args.mode,
    warmup_epochs=5, # set to 1 or 2 or this is okay?
    device=device,
    return_models=True
)

plot_train_history(history, f'plots/hidden-function/train-val-loss-score/{args.cycles}cycles_{args.mode}{"_corr" if args.corr else ""}')

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
FPS = 5
models = [model_0] + history['models']

def plot_cosines(i):
    cosines = get_cosine_angles(models[i])
    ax.clear()

    ax.scatter(range(len(cosines)), cosines.abs())
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel('Absolute value of cosine')
    ax.set_xlabel('Weight number')
    ax.set_title(f'Epoch {i}')

ani = FuncAnimation(fig, plot_cosines, frames = len(models), interval=1000/FPS, repeat=True)
writer = FFMpegWriter(fps=FPS, bitrate=1800, metadata=dict(arist='Me'))

ani.save(f'animations/cosine_variations/{args.cycles}cycles_{args.mode}{"_corr" if args.corr else ""}', writer=writer)
"""
IS THE ANIMATION REALLY REQUIRED WHEN WE ARE USING WARMUP TRAINING METHODS? I DON'T THINK SO
"""

preds_train_nn, preds_val_nn, preds_test_nn = predict(best_model_nn, X_train, device), predict(best_model_nn, X_val, device), predict(best_model_nn, X_test, device)
add_log(f'preds_train_nn.shape == {preds_train_nn.shape}, preds_val_nn.shape == {preds_val_nn.shape}, preds_test_nn.shape == {preds_test_nn.shape}')

score_train, score_val, score_test = metric(preds_train_nn, Y_train), metric(preds_val_nn, Y_val), metric(preds_test_nn, Y_test)
add_log(f'score_train == {score_train}, score_val == {score_val}, score_test == {score_test}')

plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.scatter(X_train.sum(dim=1), Y_train, c='g', label='True', s=4, alpha=0.2)
plt.scatter(X_train.sum(dim=1), preds_train_nn, c='r', label='Predicted', s=4, alpha=0.2)
plt.title('Train')
plt.xlim(-args.cycles*torch.pi/scale_factor, args.cycles*torch.pi/scale_factor)
plt.ylim(-1.25, 1.25)
plt.legend()

plt.subplot(122)
plt.scatter(X_test.sum(dim=1), Y_test, c='g', label='True', s=4, alpha=0.2)
plt.scatter(X_test.sum(dim=1), preds_test_nn, c='r', label='Predicted', s=4, alpha=0.2)
plt.title('Test')
plt.xlim(-args.cycles*torch.pi/scale_factor, args.cycles*torch.pi/scale_factor)
plt.ylim(-1.25, 1.25)
plt.legend()

plt.suptitle('NN')
plt.savefig(f'plots/hidden-function/nn-fit/{args.cycles}cycles_{args.mode}{"_corr" if args.corr else ""}')

"""
SVM CROSS-VAL AND TRAINING
"""

X_train_ntk = get_ntk_feature_matrix(X_train, best_model_nn)
X_val_ntk = get_ntk_feature_matrix(X_val, best_model_nn)

model_base_ntk = LinearSVR()
params_ntk = {
    'C': np.logspace(-4, 2, 7)
}

gammas = np.logspace(-4, 4, 9).tolist()
gammas.append('scale')
gammas.append('auto')
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
    best_model_km.fit(X_train_ntk, Y_train)
    preds_train_km, preds_val_km, preds_test_km = best_model_km.predict(X_train_ntk), best_model_km.predict(X_val_ntk), best_model_km.predict(X_test_ntk)
else:
    best_model_km = SVR(C=best_params_rbf['C'], gamma=best_params_rbf['gamma'])
    best_model_km.fit(X_train, Y_train)
    preds_train_km, preds_val_km, preds_test_km = best_model_km.predict(X_train), best_model_km.predict(X_val), best_model_km.predict(X_test)

add_log(f'preds_train_km.shape == {preds_train_km.shape}, preds_val_km.shape == {preds_val_km.shape}, preds_test_km.shape == {preds_test_km.shape}')

score_train, score_val, score_test = mean_squared_error(Y_train, preds_train_km), mean_squared_error(Y_val, preds_val_km), mean_squared_error(Y_test, preds_test_km)
add_log(f'score_train == {score_train}, score_val == {score_val}, score_test == {score_test}')

plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.scatter(X_train.sum(dim=1), Y_train, c='g', label='True', s=4, alpha=0.2)
plt.scatter(X_train.sum(dim=1), preds_train_km, c='r', label='Predicted', s=4, alpha=0.2)
plt.title('Train')
plt.xlim(-args.cycles*torch.pi/scale_factor, args.cycles*torch.pi/scale_factor)
plt.ylim(-1.25, 1.25)
plt.legend()

plt.subplot(122)
plt.scatter(X_test.sum(dim=1), Y_test, c='g', label='True', s=4, alpha=0.2)
plt.scatter(X_test.sum(dim=1), preds_test_km, c='r', label='Predicted', s=4, alpha=0.2)
plt.title('Test')
plt.xlim(-args.cycles*torch.pi/scale_factor, args.cycles*torch.pi/scale_factor)
plt.ylim(-1.25, 1.25)
plt.legend()

plt.suptitle(f'SVM ({best_kernel})')
plt.savefig(f'plots/hidden-function/svm-fit/{args.cycles}cycles_{args.mode}{"_corr" if args.corr else ""}')
