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
parser.add_argument('-deg', type=int, default=4, help='Degree of polynomial to generate')
parser.add_argument('-corr', action='store_true', help='Whether to draw data from distribution with non-identity covariance')
parser.add_argument('-eps', type=json.loads, help='Epochs for which to plot NTK dynamics')
parser.add_argument('-kernel', type=str, choices=['rbf', 'ntk'], default='ntk', help='SVM kernel to train')
parser.add_argument('-C', type=float, help='C to use for training SVM')
parser.add_argument('-gamma', type=float, default=None, help='gamma to use for training RBF kernel')

args = parser.parse_args()
if args.gamma is None:
    GAMMA = 'scale'
else:
    GAMMA = args.gamma

if not Path('plots/hidden-func-align/').is_dir():
    if not Path('plots/').is_dir():
        os.mkdir('plots/')
    os.mkdir('plots/hidden-func-align/')
    os.mkdir('plots/hidden-func-align/alignments/')
    os.mkdir('plots/hidden-func-align/partial-fits/')
    os.mkdir('plots/hidden-func-align/nn-fit/')
    os.mkdir('plots/hidden-func-align/svm-fit/')

plot_dir = 'plots/hidden-func-align/'
plot_file_name = f'degree_{args.deg}{"_corr" if args.corr else ""}'
nn_fit_plot_path = plot_dir + 'nn-fit/' + plot_file_name + '.png'
svm_fit_plot_path = plot_dir + 'svm-fit/' + plot_file_name + '.png'
alignment_path = plot_dir + 'alignments/' + plot_file_name
partial_fit_path = plot_dir + 'partial-fits/' + plot_file_name

DATA_SIZE = 10000
NUM_FEATURES = 100
RANGE = 2
Y_LIM = (-3, 3)

class SimpleNN(nn.Module):
    def __init__(self, width, symmetric_init=True) -> None:
        super().__init__()
        self.width = width
        self.input = nn.Sequential(
            nn.Linear(in_features=NUM_FEATURES, out_features=width),
            nn.ReLU()
        )
        if symmetric_init:
            for param in self.parameters():
                n = len(param)
                first_ws = param[:n//2].detach()
                param.data=torch.cat((first_ws, -first_ws))
        self.output = nn.Sequential(
            nn.Linear(in_features=width, out_features=1)
        )

    def forward(self, x):
        return self.output(self.input(x))
    
    def clone(self):
        new_model = SimpleNN(self.width)
        new_model.load_state_dict(self.state_dict())
        return new_model

    def freeze_first(self) -> None:
        for param in self.input.parameters():
            param.requires_grad = False
        
    def freeze_last(self) -> None:
        for param in self.output.parameters():
            param.requires_grad_(False)
    
    def revive_last(self) -> None:
        for param in self.output.parameters():
            param.requires_grad_(True)
    
    def reinit_last(self) -> None:
        for child in self.output.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

extra_cov = 1.0 if args.corr else 0.0

distribution = MultivariateNormal(torch.zeros(NUM_FEATURES), torch.eye(NUM_FEATURES) + extra_cov)

torch.manual_seed(844) 
X = distribution.sample((DATA_SIZE,))

scale_factor = RANGE / X.sum(dim=1).abs().max()
X *= scale_factor
roots = torch.linspace(-RANGE, RANGE, args.deg)
X_sum = X.sum(dim=1)
Y = torch.ones_like(X_sum)
for root in roots:
    Y *= (X_sum - root)
Y.unsqueeze_(1)

X_training, X_test, Y_training, Y_test = train_test_split(X, Y, test_size=0.1, random_state=13)
X_train, X_val, Y_train, Y_val = train_test_split(X_training, Y_training, test_size=1/9, random_state=9173)

train_dataloader, val_dataloader = make_dataloader(X_train, Y_train, 32, True), make_dataloader(X_val, Y_val, 32, True)

device = 'cpu'

def get_cosine_angles(model: SimpleNN, ref_vector: torch.Tensor = torch.ones(NUM_FEATURES)):
    weights = next(model.input.parameters()).detach()
    return torch.tensor([torch.dot(weight, ref_vector) / (torch.linalg.norm(ref_vector) * torch.linalg.norm(weight)) for weight in weights])

config_file = Path(f'configs/nn/hidden-func-poly/{"corr_" if args.corr else ""}degree_{args.deg}.json')
with open(config_file, 'r') as f:
    best_config = json.load(f)
best_width = best_config['width']
best_eta = best_config['eta']
best_weight_decay = best_config['weight_decay']
best_score = best_config['score']

torch.random.manual_seed(47647)
best_model_nn = SimpleNN(best_width).to(device)
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
    device=device,
    return_models=True
)

models = [model_0] + history['models']

preds_train_nn, preds_val_nn, preds_test_nn = predict(best_model_nn, X_train, device), predict(best_model_nn, X_val, device), predict(best_model_nn, X_test, device)
score_train, score_val, score_test = metric(preds_train_nn, Y_train), metric(preds_val_nn, Y_val), metric(preds_test_nn, Y_test)

plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.scatter(X_train.sum(dim=1), Y_train, c='g', label='True', s=4, alpha=0.2)
plt.scatter(X_train.sum(dim=1), preds_train_nn, c='r', label='Predicted', s=4, alpha=0.2)
plt.title('Train')
plt.xlim(-RANGE, RANGE)
plt.ylim(Y_LIM)
plt.legend()

plt.subplot(122)
plt.scatter(X_test.sum(dim=1), Y_test, c='g', label='True', s=4, alpha=0.2)
plt.scatter(X_test.sum(dim=1), preds_test_nn, c='r', label='Predicted', s=4, alpha=0.2)
plt.title('Test')
plt.xlim(-RANGE, RANGE)
plt.ylim(Y_LIM)
plt.legend()

plt.suptitle('NN')
plt.savefig(nn_fit_plot_path)

model_base_ntk = LinearSVR()
params_ntk = {
    'C': np.logspace(-4, 2, 7)
}
scorer = make_scorer(mean_squared_error, greater_is_better=False)
model_cv_ntk = GridSearchCV(
    estimator=model_base_ntk,
    param_grid=params_ntk,
    scoring=scorer,
    n_jobs=1,
    cv=4,
    refit=False
)

for epoch in args.eps:
    print(f'Beginning epoch {epoch}')
    file_name = f'_epoch_{epoch}.png'
    X_train_ntk = get_ntk_feature_matrix(X_train, models[epoch])
    X_val_ntk = get_ntk_feature_matrix(X_val, models[epoch])
    X_test_ntk = get_ntk_feature_matrix(X_test, models[epoch])
    model_cv_ntk.fit(X_train_ntk, Y_train.squeeze())
    best_params_ntk = model_cv_ntk.best_params_
    best_score_ntk = max(model_cv_ntk.cv_results_['mean_test_score'])
    print(f'Best C: {best_params_ntk["C"]}, best score: {best_score_ntk}')
    best_model_km = LinearSVR(C=best_params_ntk['C'])
    best_model_km.fit(X_train_ntk, Y_train.squeeze())
    preds_train_km, preds_val_km, preds_test_km = best_model_km.predict(X_train_ntk), best_model_km.predict(X_val_ntk), best_model_km.predict(X_test_ntk)
    score_train, score_val, score_test = mean_squared_error(Y_train.squeeze(), preds_train_km), mean_squared_error(Y_val.squeeze(), preds_val_km), mean_squared_error(Y_test.squeeze(), preds_test_km)

    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.scatter(X_train.sum(dim=1), Y_train, c='g', label='True', s=4, alpha=0.2)
    plt.scatter(X_train.sum(dim=1), preds_train_km, c='r', label='Predicted', s=4, alpha=0.2)
    plt.title('Train')
    plt.xlim(-RANGE, RANGE)
    plt.ylim(Y_LIM)
    plt.legend()

    plt.subplot(122)
    plt.scatter(X_test.sum(dim=1), Y_test, c='g', label='True', s=4, alpha=0.2)
    plt.scatter(X_test.sum(dim=1), preds_test_km, c='r', label='Predicted', s=4, alpha=0.2)
    plt.title('Test')
    plt.xlim(-RANGE, RANGE)
    plt.ylim(Y_LIM)
    plt.legend()

    plt.suptitle(f'NTK (epoch {epoch})')
    plt.savefig(f'{partial_fit_path}{file_name}')

    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    cosines = get_cosine_angles(models[epoch])
    plt.scatter(range(len(cosines)), cosines.abs())
    plt.ylim(-0.1, 1.1)
    plt.ylabel('Cosine similarity between weights and ones')
    plt.xlabel('Weight number')
    plt.title('Cosine similarity/alignment')

    plt.subplot(122)
    plt.scatter(preds_train_nn.squeeze(), preds_train_km, c='r', label='Train', s=4, alpha=0.2)
    plt.scatter(preds_test_nn.squeeze(), preds_test_km, c='g', label='Test', s=4, alpha=0.2)
    plt.xlabel('NN preds')
    plt.ylabel('NTK preds')
    plt.ylim(Y_LIM)
    plt.xlim(Y_LIM)
    plt.legend()
    plt.title('Prediction correlations of NN and NTK')
    
    plt.suptitle(f'Alignment, epoch {epoch}')
    plt.savefig(f'{alignment_path}{file_name}')

"""
SVM CROSS-VAL AND TRAINING
"""

X_train_ntk = get_ntk_feature_matrix(X_train, best_model_nn)
X_val_ntk = get_ntk_feature_matrix(X_val, best_model_nn)
if args.kernel == 'ntk':
    X_test_ntk = get_ntk_feature_matrix(X_test, best_model_nn)
    best_model_km = LinearSVR(C=args.C)
    best_model_km.fit(X_train_ntk, Y_train.squeeze())
    preds_train_km, preds_val_km, preds_test_km = best_model_km.predict(X_train_ntk), best_model_km.predict(X_val_ntk), best_model_km.predict(X_test_ntk)
else:
    best_model_km = SVR(C=args.C, gamma=GAMMA)
    best_model_km.fit(X_train, Y_train.squeeze())
    preds_train_km, preds_val_km, preds_test_km = best_model_km.predict(X_train), best_model_km.predict(X_val), best_model_km.predict(X_test)

score_train, score_val, score_test = mean_squared_error(Y_train.squeeze(), preds_train_km), mean_squared_error(Y_val.squeeze(), preds_val_km), mean_squared_error(Y_test.squeeze(), preds_test_km)

plt.figure(figsize=(12, 6)) 

plt.subplot(121)
plt.scatter(X_train.sum(dim=1), Y_train, c='g', label='True', s=4, alpha=0.2)
plt.scatter(X_train.sum(dim=1), preds_train_km, c='r', label='Predicted', s=4, alpha=0.2)
plt.title('Train')
plt.xlim(-RANGE, RANGE)
plt.ylim(Y_LIM)
plt.legend()

plt.subplot(122)
plt.scatter(X_test.sum(dim=1), Y_test, c='g', label='True', s=4, alpha=0.2)
plt.scatter(X_test.sum(dim=1), preds_test_km, c='r', label='Predicted', s=4, alpha=0.2)
plt.title('Test')
plt.xlim(-RANGE, RANGE)
plt.ylim(Y_LIM)
plt.legend()

plt.suptitle(f'SVM ({args.kernel})')
plt.savefig(svm_fit_plot_path)
