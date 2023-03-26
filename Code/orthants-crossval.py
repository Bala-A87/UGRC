import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
from pathlib import Path
from scripts.data.orthants import generate_train_data, generate_test_data, ORTHANTS, generate_point
from scripts.utils import make_dataloader, EarlyStopping, plot_train_history
from scripts.models import SimpleNN
from scripts.train import train_model
from scripts.metrics import BinaryAccuracy
from scripts.ntk import get_ntk_feature_matrix
from scripts.test import predict
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, make_scorer

def add_log(log_str: str, file_name: str):
    with open(file_name, 'a') as f:
        f.write(log_str)

log_path = Path('logs/orthants-single-empty/')
if not log_path.is_dir():
    if not Path('logs/').is_dir():
        os.mkdir('logs/')
    os.mkdir(log_path)

device = 'cpu'

args_parser = ArgumentParser()
args_parser.add_argument('-d', '--depth', type=int, help='Depth (hidden layers) of the neural network')
args_parser.add_argument('-c', '--centre', type=float, default=4., help='Centre of the sphere = CENTRE * torch.ones(7)')

args = args_parser.parse_args()
config = {
    'depth': args.depth,
    'widths': [32, 64, 128],
    'etas': [1e-4, 1e-3, 1e-2],
    'weight_decays': np.logspace(-5, 5, 11).tolist() + [0.0]
}

plot_path_str = f'plots/orthants-single-empty/centre_{args.centre}-{config["depth"]}/'
plot_path = Path(plot_path_str)
if not plot_path.is_dir():
    if not Path('plots/orthants-single-empty/').is_dir():
        if not Path('plots/').is_dir():
            os.mkdir('plots/')
        os.mkdir('plots/orthants-single-empty/')
    os.mkdir(plot_path)

log_file = f'logs/orthants-single-empty/centre_{args.centre}-{config["depth"]}.txt'
if Path(log_file).is_file():
    os.remove(log_file)

HIGH_COUNT = 100
LOW_FRAC = 1/64
ZERO_FRAC = 0.5
TEST_COUNT = 100
CENTRE = args.centre * torch.ones(7) 
LOW_RADIUS = 1.
HIGH_RADIUS = 2.

X_train, Y_train, orthant_counts = generate_train_data(
    low_count=HIGH_COUNT,
    high_count=HIGH_COUNT,
    low_spread=0,
    high_spread=0,
    low_frac=LOW_FRAC,
    zero_frac=ZERO_FRAC,
    centre=CENTRE,
    random_state=535
)
add_log(f'X_train.shape == {X_train.shape}\n', log_file)
add_log(f'Y_train.shape == {Y_train.shape}\n', log_file)

for i in range(128):
    if orthant_counts[i] == 0:
        ZERO_ORTHANT_INDEX = i
add_log(f'Empty orthant is orthant number {ZERO_ORTHANT_INDEX}\n', log_file)

X_test, Y_test = generate_test_data(TEST_COUNT, CENTRE, random_state=652)
add_log(f'X_test.shape == {X_test.shape}\n', log_file)
add_log(f'Y_test.shape == {Y_test.shape}\n', log_file)

X_val, Y_val = X_test[ZERO_ORTHANT_INDEX], Y_test[ZERO_ORTHANT_INDEX]

train_dataloader, val_dataloader = make_dataloader(X_train, Y_train, 32, True), make_dataloader(X_val, Y_val, 32, True)

X_total_0 = X_train[Y_train.squeeze()==0]
X_total_1 = X_train[Y_train.squeeze()==1]

neighboring_orthants = []
for i in range(128):
    if (ORTHANTS[i] * ORTHANTS[ZERO_ORTHANT_INDEX]).sum() == 5:
        neighboring_orthants.append(i)
add_log(f'Orthants neighboring the empty orthant are: {neighboring_orthants}\n', log_file)

key_orthants = [ZERO_ORTHANT_INDEX, neighboring_orthants[0], list(set(range(128)) - set(neighboring_orthants+[ZERO_ORTHANT_INDEX]))[0]]
key_orthants_types = ['empty', 'neighbor', 'random']

X_empty_0 = X_val[Y_val.squeeze() == 0]
X_empty_1 = X_val[Y_val.squeeze() == 1]
add_log(f'X_empty_0.shape == {X_empty_0.shape}\n', log_file)
add_log(f'X_empty_1.shape == {X_empty_1.shape}\n', log_file)

widths = config["widths"]
etas = config["etas"]
weight_decays = config["weight_decays"]
best_width = None
best_eta = None
best_weight_decay = None
best_score = -torch.inf
total_count = len(widths) * len(etas) * len(weight_decays)
curr_count = 0
EPOCHS = 10

add_log(f'Cross-validating across {total_count} models.\n\n', log_file)

for width in widths:
    for eta in etas:
        for weight_decay in weight_decays:
            model = SimpleNN(7, hidden_layers=config["depth"], hidden_units=width).to(device)
            loss_fn = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=eta, weight_decay=weight_decay)
            metric = BinaryAccuracy()

            history = train_model(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                metric=metric,
                epochs=EPOCHS,
                verbose=0,
                device=device
            )
            curr_count += 1
            score = history['val_score'][-1]
            add_log(f'[{curr_count}/{total_count}] Width: {width}, lr: {eta}, lambda: {weight_decay} ==> score: {score:.6f}\n', log_file)
            if score > best_score:
                best_score = score
                best_width = width
                best_eta = eta
                best_weight_decay = weight_decay

add_log(f'\nBest validation score after {EPOCHS} epochs: {best_score:.6f}\n', log_file)
add_log(f'Best configuration: width: {best_width}, lr: {best_eta}, lambda: {best_weight_decay}\n', log_file)

best_model_nn = SimpleNN(7, hidden_layers=config["depth"], hidden_units=best_width).to(device)
model_0 = best_model_nn.clone()
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(params=best_model_nn.parameters(), lr=best_eta, weight_decay=best_weight_decay)
metric = BinaryAccuracy()
early_stop = EarlyStopping(patience=20, min_delta=1e-4)

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

plot_train_history(history, save_file=plot_path_str+'train_val_loss_score.png')

def get_output_layer_features(model: SimpleNN, u: torch.Tensor) -> torch.Tensor:
    return model.hidden(model.input(u)).detach()

def get_output_layer_feature_matrix(model: SimpleNN, U: torch.Tensor) -> torch.Tensor:
    return torch.cat([get_output_layer_features(model, u).reshape(1, -1) for u in U])

def get_min_dists_inds(x: torch.Tensor, Y: torch.Tensor, labels: torch.Tensor, k: int = 10):
    dists = ((x.reshape(1, -1) - Y)**2).sum(dim=1).sqrt().sort()
    top_inds = dists[1][:k]
    top_dists = dists[0][:k]
    top_classes = torch.tensor([labels[ind].squeeze() for ind in top_inds])
    return torch.cat([top_dists.reshape(-1, 1), top_classes.reshape(-1, 1)], dim=1)

def get_all_min_dists_inds(X: torch.Tensor, Y: torch.Tensor, labels: torch.Tensor, k: int = 10):
    return torch.cat([get_min_dists_inds(x, Y, labels, k).unsqueeze(0) for x in X])

COLORS = np.array(['r', 'g'])

def plot_closest_distances(
    model_index: int = -1,
    k: int = 10
) -> None:
    model = models[model_index]

    top_dists_eucl_0, top_dists_eucl_1 = get_all_min_dists_inds(X_empty_0, X_train, Y_train, k), get_all_min_dists_inds(X_empty_1, X_train, Y_train, k)

    top_dists_last_0, top_dists_last_1 = get_all_min_dists_inds(get_output_layer_feature_matrix(model, X_empty_0), get_output_layer_feature_matrix(model, X_train), Y_train, k), get_all_min_dists_inds(get_output_layer_feature_matrix(model, X_empty_1), get_output_layer_feature_matrix(model, X_train), Y_train, k)

    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    for i in range(TEST_COUNT): 
        ax[0][0].scatter(i*torch.ones(k), top_dists_eucl_0[i][:, 0], c=COLORS[top_dists_eucl_0[i][:, 1].numpy().astype(int)], s=10, alpha=0.4)
        ax[0][0].set_title('Eucl, neg')
    for i in range(TEST_COUNT):
        ax[0][1].scatter(i*torch.ones(k), top_dists_eucl_1[i][:, 0], c=COLORS[top_dists_eucl_1[i][:, 1].numpy().astype(int)], s=10, alpha=0.4)
        ax[0][1].set_title('Eucl, pos')
    for i in range(TEST_COUNT):
        ax[1][0].scatter(i*torch.ones(k), top_dists_last_0[i][:, 0], c=COLORS[top_dists_last_0[i][:, 1].numpy().astype(int)], s=10, alpha=0.4)
        ax[1][0].set_title('Last, neg')
    for i in range(TEST_COUNT):
        ax[1][1].scatter(i*torch.ones(k), top_dists_last_1[i][:, 0], c=COLORS[top_dists_last_1[i][:, 1].numpy().astype(int)], s=10, alpha=0.4)
        ax[1][1].set_title('Last, pos')

    plt.suptitle(f'Epoch {model_index}, red: neg, green: pos')
    plt.savefig(f'{plot_path_str}dists_{"init" if model_index==0 else "final"}_k_{k}.png')

plot_closest_distances(0, 5)
plot_closest_distances(-1, 5)
plot_closest_distances(0, 10)
plot_closest_distances(-1, 10)
plot_closest_distances(0, 20)
plot_closest_distances(-1, 20)

scores_nn = torch.tensor([
    metric(
        predict(best_model_nn, X_test[i], device),
        Y_test[i]
    ) for i in range(128)
])

add_log(f'Average accuracy across all orthants: {scores_nn.mean()}\n', log_file)
add_log(f'Accuracy in empty orthant: {scores_nn[ZERO_ORTHANT_INDEX].mean()}\n', log_file)

X_train_ntk = get_ntk_feature_matrix(X_train, best_model_nn)
X_val_ntk = get_ntk_feature_matrix(X_val, best_model_nn)

Cs = np.logspace(-5, 5, 11)
best_C = None
best_score = -torch.inf
scorer = make_scorer(accuracy_score)
total_count = len(Cs)
curr_count = 0
add_log(f'Cross-validation across {total_count} models.\n\n', log_file)
for C in Cs:
    curr_count += 1
    model_cv = LinearSVC(C=C)
    model_cv.fit(X_train_ntk, Y_train.squeeze())
    preds_train, preds_val = model_cv.predict(X_train_ntk), model_cv.predict(X_val_ntk)
    score_train, score_val = accuracy_score(Y_train.squeeze(), preds_train), accuracy_score(Y_val.squeeze(), preds_val)
    if score_val > best_score:
        best_C = C
        best_score = score_val
    add_log(f'[{curr_count}/{total_count}]\tC:{C}, train score:{score_train}, val score:{score_val}\n', log_file)
add_log(f'\nBest validation accuracy: {best_score}, for C = {best_C}\n', log_file)

best_model_km = LinearSVC(C=best_C, max_iter=int(2e4))
best_model_km.fit(X_train_ntk, Y_train.squeeze())

preds_train, preds_val = best_model_km.predict(X_train_ntk), best_model_km.predict(X_val_ntk)
score_train, score_val = accuracy_score(Y_train.squeeze(), preds_train), accuracy_score(Y_val.squeeze(), preds_val)
add_log(f'Train accuracy with NTK: {score_train}\n', log_file)
add_log(f'Validation accuracy with NTK: {score_val}\n', log_file)

scores_km = np.array([
    accuracy_score(
        best_model_km.predict(get_ntk_feature_matrix(X_test[i], best_model_nn)),
        Y_test[i].squeeze()
    ) for i in range(128)
])

add_log(f'Average accuracy across all orthants: {scores_km.mean()}\n', log_file)
add_log(f'Accuracy in empty orthant: {scores_km[ZERO_ORTHANT_INDEX].mean()}\n', log_file)


plt.figure(figsize=(13, 6)) 

plt.subplot(121)
plt.scatter(orthant_counts, scores_nn)
plt.xlabel('Number of points')
plt.ylabel('Binary accuracy')
plt.ylim((0., 1.))
plt.title('NN')

plt.subplot(122)
plt.scatter(orthant_counts, scores_km)
plt.xlabel('Number of points')
plt.ylabel('Binary accuracy')
plt.ylim((0., 1.))
plt.title('SVM')

plt.suptitle('Accuracy per orthant')

plt.savefig(plot_path_str+'accuracy_vs_counts.png')

c_arr = np.zeros(128)
c_arr[ZERO_ORTHANT_INDEX] = 2
for i in neighboring_orthants:
    c_arr[i] = 1
non_neighboring_orthants = list(set(range(128)) - set([ZERO_ORTHANT_INDEX]+neighboring_orthants))

plt.figure(figsize=(16, 8))

plt.subplot(121)
plt.scatter([ZERO_ORTHANT_INDEX], [scores_nn[ZERO_ORTHANT_INDEX]], c='g', s=25, label='Empty')
plt.scatter(neighboring_orthants, scores_nn[neighboring_orthants], c='r', s=15, label='Neighbors')
plt.scatter(non_neighboring_orthants, scores_nn[non_neighboring_orthants], c='b', s=5, label='Non-neighbors')
plt.legend()
plt.ylim(-0.05, 1.05)
plt.title('NN')

plt.subplot(122)
plt.scatter([ZERO_ORTHANT_INDEX], [scores_km[ZERO_ORTHANT_INDEX]], c='g', s=25, label='Empty')
plt.scatter(neighboring_orthants, scores_km[neighboring_orthants], c='r', s=15, label='Neighbors')
plt.scatter(non_neighboring_orthants, scores_km[non_neighboring_orthants], c='b', s=5, label='Non-neighbors')
plt.legend()
plt.ylim(-0.05, 1.05)
plt.title("SVM (ntk)")

plt.suptitle('Test accuracy vs orthant')

plt.savefig(plot_path_str+'accuracy_vs_orthant_number.png')

X_radial = torch.cat([torch.cat([generate_point(r, CENTRE, torch.ones(7)).unsqueeze(0) for _ in range(100)]).unsqueeze(0) for r in np.arange(0.5, 2.6, 0.1)])

plt.figure(figsize=(20, 6))
for i in range(3):
    plt.subplot(1, 3, i+1)
    X_dec_bound = X_radial * ORTHANTS[key_orthants[i]]
    proba_orthants = torch.tensor([predict(best_model_nn, X_r, device).mean() for X_r in X_dec_bound])
    plt.plot(np.arange(0.5, 2.6, 0.1), proba_orthants)
    plt.ylim(0.0, 1.0)
    plt.title(key_orthants_types[i])
plt.suptitle('Average predicted probability of class 1 vs radius, NN')
plt.savefig(plot_path_str+'nn_radial_decision_boundary.png')

plt.figure(figsize=(20, 6))
for i in range(3):
    plt.subplot(1, 3, i+1)
    X_dec_bound = X_radial * ORTHANTS[key_orthants[i]]
    proba_orthants = torch.tensor([np.mean(best_model_km.predict(get_ntk_feature_matrix(X_r, best_model_nn))) for X_r in X_dec_bound])
    plt.plot(np.arange(0.5, 2.6, 0.1), proba_orthants)
    plt.ylim(0.0, 1.0)
    plt.title(key_orthants_types[i])
plt.suptitle('Average predicted probability (estimated) of class 1 vs radius, SVM')
plt.savefig(plot_path_str+'svm_radial_decision_boundary.png')
