import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
from pathlib import Path
from scripts.data.orthants import generate_train_data, generate_test_data, ORTHANTS, generate_point
from scripts.utils import make_dataloader, EarlyStopping
from scripts.models import SimpleNN
from scripts.train import train_model
from scripts.metrics import BinaryAccuracy
from scripts.ntk import get_ntk_feature_matrix
from scripts.test import predict
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV

log_path = Path('logs/orthants-final/')
if not log_path.is_dir():
    if not Path('logs/').is_dir():
        os.mkdir('logs/')
    os.mkdir(log_path)

device = 'cpu'

args_parser = ArgumentParser()
args_parser.add_argument('-d', '--depth', type=int, help='Depth (hidden layers) of the neural network')
args_parser.add_argument('-p', '--pure', action='store_true', help='Whether to run for pure (rbf, linear) svms as well')

args = args_parser.parse_args()

plot_path_str = f'plots/orthants-final/depth_{args.depth}/'
plot_path = Path(plot_path_str)
if not plot_path.is_dir():
    if not Path('plots/orthants-final/').is_dir():
        if not Path('plots/').is_dir():
            os.mkdir('plots/')
        os.mkdir('plots/orthants-final/')
    os.mkdir(plot_path)

log_file = f'logs/orthants-final/depth_{args.depth}.txt'
if Path(log_file).is_file():
    os.remove(log_file)

def add_log(log_str: str, file_name: str = log_file):
    with open(file_name, 'a') as f:
        f.write(log_str+'\n')

HIGH_COUNT = 100
LOW_FRAC = 1/64
ZERO_FRAC = 0.5
TEST_COUNT = 100
CENTRE = 4. * torch.ones(7) 
LOW_RADIUS = 1.
HIGH_RADIUS = 2.
RUNS = 5

params_nn = {
    'width': [32, 64, 128],
    'eta': [1e-4, 1e-3, 1e-2],
    'weight_decay': np.logspace(-4, 4, 9).tolist() + [0.0]
}
best_params_nn = {
    'width': None,
    'eta': None,
    'weight_decay': None,
    'score': -torch.inf
}

params_km_rbf = {
    'C': np.logspace(-4, 4, 9),
    'gamma': np.logspace(-4, 4, 9).tolist() + ['scale', 'auto']
}
best_params_km_rbf = {
    'C': None,
    'gamma': None,
    'score': -torch.inf
}

params_km_lin = {
    'C': np.logspace(-4, 4, 9)
}
best_params_km_lin = {
    'C': None,
    'score': -torch.inf
}

params_km_ntk = {
    'C': np.logspace(-4, 4, 9)
}
best_params_km_ntk = {
    'C': None,
    'score': -torch.inf
}

def get_output_layer_features(model: SimpleNN, u: torch.Tensor) -> torch.Tensor:
    return model.hidden(model.input(u)).detach()

def get_output_layer_feature_matrix(model: SimpleNN, U: torch.Tensor) -> torch.Tensor:
    return torch.cat([get_output_layer_features(model, u).reshape(1, -1) for u in U])

def get_min_dist_class(x: torch.Tensor, Y: torch.Tensor, labels: torch.Tensor):
    dists = ((x.reshape(1, -1) - Y)**2).sum(dim=1).sqrt()
    top_ind = dists.argmin()
    top_class = labels[top_ind].reshape(1,)
    return top_class

def get_all_min_dist_classes(X: torch.Tensor, Y: torch.Tensor, labels: torch.Tensor):
    return torch.cat([get_min_dist_class(x, Y, labels) for x in X])


X_training, Y_training, orthant_counts = generate_train_data(
    low_count=HIGH_COUNT,
    high_count=HIGH_COUNT,
    low_spread=0,
    high_spread=0,
    low_frac=LOW_FRAC,
    zero_frac=ZERO_FRAC,
    centre=CENTRE
)
X_train, X_val, Y_train, Y_val = train_test_split(X_training, Y_training, test_size=0.2)

for i in range(128):
    if orthant_counts[i] == 0:
        ZERO_ORTHANT_INDEX = i
add_log(f'Empty orthant is orthant number {ZERO_ORTHANT_INDEX}')

X_test, Y_test = generate_test_data(TEST_COUNT, CENTRE)

train_dataloader, val_dataloader = make_dataloader(X_train, Y_train, 32, True), make_dataloader(X_val, Y_val, 32, True)

X_total_0 = X_train[Y_train.squeeze()==0]
X_total_1 = X_train[Y_train.squeeze()==1]

neighboring_orthants = []
for i in range(128):
    if (ORTHANTS[i] * ORTHANTS[ZERO_ORTHANT_INDEX]).sum() == 5:
        neighboring_orthants.append(i)
add_log(f'Orthants neighboring the empty orthant are: {neighboring_orthants}')

key_orthants = [ZERO_ORTHANT_INDEX, neighboring_orthants[0], list(set(range(128)) - set(neighboring_orthants+[ZERO_ORTHANT_INDEX]))[0]]
key_orthants_types = ['empty', 'neighbor', 'random']

X_empty = X_test[ZERO_ORTHANT_INDEX]
Y_empty = Y_test[ZERO_ORTHANT_INDEX]
X_empty_0 = X_empty[Y_empty.squeeze() == 0]
X_empty_1 = X_empty[Y_empty.squeeze() == 1]

radial_range = np.arange(0.5, 2.6, 0.1)
X_radial = torch.cat([torch.cat([generate_point(r, CENTRE, torch.ones(7)).unsqueeze(0) for _ in range(100)]).unsqueeze(0) for r in radial_range])

orthant_scores_nn = np.zeros((RUNS,)).tolist()
orthant_scores_km_ntk = np.zeros((RUNS,)).tolist()
orthant_scores_km_rbf = np.zeros((RUNS,)).tolist()
orthant_scores_km_lin = np.zeros((RUNS,)).tolist()
match_fracs_0_init = np.zeros((RUNS,)).tolist()
match_fracs_1_init = np.zeros((RUNS,)).tolist()
match_fracs_0_final = np.zeros((RUNS,)).tolist()
match_fracs_1_final = np.zeros((RUNS,)).tolist()
rad_probas_nn = np.zeros((len(key_orthants), RUNS)).tolist()
rad_probas_km_ntk = np.zeros((len(key_orthants), RUNS)).tolist()
rad_probas_km_rbf = np.zeros((len(key_orthants), RUNS)).tolist()
rad_probas_km_lin = np.zeros((len(key_orthants), RUNS)).tolist()


for run in range(RUNS):
    add_log('\n' + '*'*100)
    add_log(f'Run number {run}')

    if best_params_nn['width'] is None:
        total_count = len(params_nn['width']) * len(params_nn['eta']) * len(params_nn['weight_decay'])
        curr_count = 0
        EPOCHS = 10

        add_log(f'Cross-validating across {total_count} models.\n')

        for width in params_nn['width']:
            for eta in params_nn['eta']:
                for weight_decay in params_nn['weight_decay']:
                    model = SimpleNN(7, hidden_layers=args.depth, hidden_units=width).to(device)
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
                    add_log(f'[{curr_count}/{total_count}] Width: {width}, lr: {eta}, lambda: {weight_decay} ==> score: {score:.6f}')
                    if score > best_params_nn['score']:
                        best_params_nn['score'] = score
                        best_params_nn['width'] = width
                        best_params_nn['eta'] = eta
                        best_params_nn['weight_decay'] = weight_decay

        add_log(f'\nBest validation score after {EPOCHS} epochs: {best_params_nn["score"]:.6f}')
        add_log(f'Best configuration: width: {best_params_nn["width"]}, lr: {best_params_nn["eta"]}, lambda: {best_params_nn["weight_decay"]}')

    best_model_nn = SimpleNN(7, hidden_layers=args.depth, hidden_units=best_params_nn['width']).to(device)
    model_0 = best_model_nn.clone()
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=best_model_nn.parameters(), lr=best_params_nn['eta'], weight_decay=best_params_nn['weight_decay'])
    metric = BinaryAccuracy()
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

    # Find the fraction of points matching the class for each class 
    closest_classes_0_init = get_all_min_dist_classes(get_output_layer_feature_matrix(model_0, X_empty_0), get_output_layer_feature_matrix(model_0, X_train), Y_train)
    closest_classes_1_init = get_all_min_dist_classes(get_output_layer_feature_matrix(model_0, X_empty_1), get_output_layer_feature_matrix(model_0, X_train), Y_train)
    closest_classes_0_final = get_all_min_dist_classes(get_output_layer_feature_matrix(best_model_nn, X_empty_0), get_output_layer_feature_matrix(best_model_nn, X_train), Y_train)
    closest_classes_1_final = get_all_min_dist_classes(get_output_layer_feature_matrix(best_model_nn, X_empty_1), get_output_layer_feature_matrix(best_model_nn, X_train), Y_train)
    match_fracs_0_init[run] = (closest_classes_0_init == 0).sum().item() / len(closest_classes_0_init)
    match_fracs_1_init[run] = (closest_classes_1_init == 1).sum().item() / len(closest_classes_1_init)
    match_fracs_0_final[run] = (closest_classes_0_final == 0).sum().item() / len(closest_classes_0_final)
    match_fracs_1_final[run] = (closest_classes_1_final == 1).sum().item() / len(closest_classes_1_final)

    scores_nn = torch.tensor([
        metric(
            predict(best_model_nn, X_test[i], device),
            Y_test[i]
        ) for i in range(128)
    ])
    
    add_log(f'Average accuracy across all orthants: {scores_nn.mean()}')
    add_log(f'Accuracy in empty orthant: {scores_nn[ZERO_ORTHANT_INDEX].mean()}')
    orthant_scores_nn[run] = scores_nn.numpy()

    for i in range(len(key_orthants)):
        X_boundary = X_radial * ORTHANTS[key_orthants[i]]
        proba_radial = torch.tensor([predict(best_model_nn, X_r, device).mean() for X_r in X_boundary])
        rad_probas_nn[i][run] = proba_radial.numpy()

    X_train_ntk = get_ntk_feature_matrix(X_train, best_model_nn)
    X_val_ntk = get_ntk_feature_matrix(X_val, best_model_nn)
    
    if best_params_km_ntk['C'] is None:
        total_count = len(params_km_ntk['C'])
        curr_count = 0
        add_log(f'Cross-validation across {total_count} models.\n')
        for C in params_km_ntk['C']:
            curr_count += 1
            model = LinearSVC(C=C)
            model.fit(X_train_ntk, Y_train.squeeze())
            preds_train, preds_val = model.predict(X_train_ntk), model.predict(X_val_ntk)
            score_train, score_val = accuracy_score(Y_train.squeeze(), preds_train), accuracy_score(Y_val.squeeze(), preds_val)
            if score_val > best_params_km_ntk['score']:
                best_params_km_ntk['C'] = C
                best_params_km_ntk['score'] = score_val
            add_log(f'[{curr_count}/{total_count}]\tC:{C}, train score:{score_train}, val score:{score_val}')
        add_log(f'\nBest validation accuracy: {best_params_km_ntk["score"]}, for C = {best_params_km_ntk["C"]}')

    best_model_km_ntk = LinearSVC(C=best_params_km_ntk['C'], max_iter=int(2e4))
    best_model_km_ntk.fit(X_train_ntk, Y_train.squeeze())
    
    scores_km_ntk = np.array([
        accuracy_score(
            best_model_km_ntk.predict(get_ntk_feature_matrix(X_test[i], best_model_nn)),
            Y_test[i].squeeze()
        ) for i in range(128)
    ])

    add_log(f'Average accuracy across all orthants: {scores_km_ntk.mean()}')
    add_log(f'Accuracy in empty orthant: {scores_km_ntk[ZERO_ORTHANT_INDEX].mean()}')
    orthant_scores_km_ntk[run] = scores_km_ntk

    for i in range(len(key_orthants)):
        X_boundary = X_radial * ORTHANTS[key_orthants[i]]
        proba_radial = np.array([np.mean(best_model_km_ntk.predict(get_ntk_feature_matrix(X_r, best_model_nn))) for X_r in X_boundary])
        rad_probas_km_ntk[i][run] = proba_radial

    if args.pure:
        if best_params_km_rbf['C'] is None:
            model_base = SVC(kernel='rbf', max_iter=int(1e4))
            scorer = make_scorer(accuracy_score)
            model_cv = GridSearchCV(estimator=model_base, param_grid=params_km_rbf, scoring=scorer, n_jobs=4, refit=False, cv=4, verbose=3)
            model_cv.fit(X_train, Y_train.squeeze())
            best_params_km_rbf['C'] = model_cv.best_params_['C']
            best_params_km_rbf['gamma'] = model_cv.best_params_['gamma']
            add_log(f'Best params for RBF: C = {best_params_km_rbf["C"]}, gamma = {best_params_km_rbf["gamma"]}')

        best_model_km_rbf = SVC(C=best_params_km_rbf['C'], kernel='rbf', gamma=best_params_km_rbf['gamma'], probability=True)
        best_model_km_rbf.fit(X_train, Y_train.squeeze())
        
        scores_km_rbf = np.array([
            accuracy_score(
                best_model_km_rbf.predict(X_test[i]),
                Y_test[i].squeeze()
            ) for i in range(128)
        ])

        add_log(f'Average accuracy across all orthants: {scores_km_rbf.mean()}')
        add_log(f'Accuracy in empty orthant: {scores_km_rbf[ZERO_ORTHANT_INDEX].mean()}')
        orthant_scores_km_rbf[run] = scores_km_rbf

        for i in range(len(key_orthants)):
            X_boundary = X_radial * ORTHANTS[key_orthants[i]]
            proba_radial = np.array([np.mean(best_model_km_rbf.predict_proba(X_r)[:, 1]) for X_r in X_boundary])
            rad_probas_km_rbf[i][run] = proba_radial

        if best_params_km_lin['C'] is None:
            model_base = SVC(kernel='linear', max_iter=int(1e4))
            scorer = make_scorer(accuracy_score)
            model_cv = GridSearchCV(estimator=model_base, param_grid=params_km_lin, scoring=scorer, n_jobs=4, refit=False, cv=4, verbose=3)
            model_cv.fit(X_train, Y_train.squeeze())
            best_params_km_lin['C'] = model_cv.best_params_['C']
            add_log(f'Best params for linear: C = {best_params_km_lin["C"]}')

        best_model_km_lin = SVC(C=best_params_km_lin['C'], kernel='linear', probability=True)
        best_model_km_lin.fit(X_train, Y_train.squeeze())
        
        scores_km_lin = np.array([
            accuracy_score(
                best_model_km_lin.predict(X_test[i]),
                Y_test[i].squeeze()
            ) for i in range(128)
        ])

        add_log(f'Average accuracy across all orthants: {scores_km_lin.mean()}')
        add_log(f'Accuracy in empty orthant: {scores_km_lin[ZERO_ORTHANT_INDEX].mean()}')
        orthant_scores_km_lin[run] = scores_km_lin

        for i in range(len(key_orthants)):
            X_boundary = X_radial * ORTHANTS[key_orthants[i]]
            proba_radial = np.array([np.mean(best_model_km_lin.predict_proba(X_r)[:, 1]) for X_r in X_boundary])
            rad_probas_km_lin[i][run] = proba_radial


# Needed things:
# total accuracies for nn, ntk, rbf, lin -> done
# acc vs count, acc vs orthant for all 4 -> done
# radial decision boundaries for all 4 -> done
# match frac for nn -> done

add_log('\n' + '*'*100)

full_orthant_scores_nn = [(np.sum(orthant_scores_nn[i]) - orthant_scores_nn[i][ZERO_ORTHANT_INDEX])/127 for i in range(RUNS)]
empty_orthant_scores_nn = [orthant_scores_nn[i][ZERO_ORTHANT_INDEX] for i in range(RUNS)]
add_log(f'Full orthant scores for nn: {full_orthant_scores_nn}')
add_log(f'Empty orthant scores for nn: {empty_orthant_scores_nn}')

full_orthant_scores_km_ntk = [(np.sum(orthant_scores_km_ntk[i]) - orthant_scores_km_ntk[i][ZERO_ORTHANT_INDEX])/127 for i in range(RUNS)]
empty_orthant_scores_km_ntk = [orthant_scores_km_ntk[i][ZERO_ORTHANT_INDEX] for i in range(RUNS)]
add_log(f'Full orthant scores for ntk: {full_orthant_scores_km_ntk}')
add_log(f'Empty orthant scores for ntk: {empty_orthant_scores_km_ntk}')

if args.pure:
    full_orthant_scores_km_rbf = [(np.sum(orthant_scores_km_rbf[i]) - orthant_scores_km_rbf[i][ZERO_ORTHANT_INDEX])/127 for i in range(RUNS)]
    empty_orthant_scores_km_rbf = [orthant_scores_km_rbf[i][ZERO_ORTHANT_INDEX] for i in range(RUNS)]
    add_log(f'Full orthant scores for rbf: {full_orthant_scores_km_rbf}')
    add_log(f'Empty orthant scores for rbf: {empty_orthant_scores_km_rbf}')

if args.pure:
    full_orthant_scores_km_lin = [(np.sum(orthant_scores_km_lin[i]) - orthant_scores_km_lin[i][ZERO_ORTHANT_INDEX])/127 for i in range(RUNS)]
    empty_orthant_scores_km_lin = [orthant_scores_km_lin[i][ZERO_ORTHANT_INDEX] for i in range(RUNS)]
    add_log(f'Full orthant scores for lin: {full_orthant_scores_km_lin}')
    add_log(f'Empty orthant scores for lin: {empty_orthant_scores_km_lin}')

non_neighboring_orthants = list(set(range(128)) - set([ZERO_ORTHANT_INDEX]+neighboring_orthants))

avg_scores_nn = np.mean(np.array(orthant_scores_nn), axis=0)
plt.figure(figsize=(6, 6))
plt.scatter(orthant_counts, avg_scores_nn)
plt.xlabel('Number of points in the orthant')
plt.ylabel('Binary accuracy')
plt.title('NN')
plt.ylim(0., 1.)
plt.savefig(plot_path_str+'acc_vs_counts_nn.png')
plt.figure(figsize=(8, 8))
plt.scatter([ZERO_ORTHANT_INDEX], [avg_scores_nn[ZERO_ORTHANT_INDEX]], c='g', s=25, label='Empty')
plt.scatter(neighboring_orthants, avg_scores_nn[neighboring_orthants], c='r', s=15, label='Neighbors')
plt.scatter(non_neighboring_orthants, avg_scores_nn[non_neighboring_orthants], c='b', s=5, label='Non-neighbors')
plt.legend()
plt.ylim(0., 1.)
plt.xlabel('Orthant number')
plt.ylabel('Binary accuracy')
plt.title('NN')
plt.savefig(plot_path_str+'acc_vs_orthant_nn.png')

avg_scores_km_ntk = np.mean(np.array(orthant_scores_km_ntk), axis=0)
plt.figure(figsize=(6, 6))
plt.scatter(orthant_counts, avg_scores_km_ntk)
plt.xlabel('Number of points in the orthant')
plt.ylabel('Binary accuracy')
plt.title('NTK SVM')
plt.ylim(0., 1.)
plt.savefig(plot_path_str+'acc_vs_counts_ntk.png')
plt.figure(figsize=(8, 8))
plt.scatter([ZERO_ORTHANT_INDEX], [avg_scores_km_ntk[ZERO_ORTHANT_INDEX]], c='g', s=25, label='Empty')
plt.scatter(neighboring_orthants, avg_scores_km_ntk[neighboring_orthants], c='r', s=15, label='Neighbors')
plt.scatter(non_neighboring_orthants, avg_scores_km_ntk[non_neighboring_orthants], c='b', s=5, label='Non-neighbors')
plt.legend()
plt.ylim(0., 1.)
plt.xlabel('Orthant number')
plt.ylabel('Binary accuracy')
plt.title('NTK SVM')
plt.savefig(plot_path_str+'acc_vs_orthant_ntk.png')

if args.pure:
    avg_scores_km_rbf = np.mean(np.array(orthant_scores_km_rbf), axis=0)
    plt.figure(figsize=(6, 6))
    plt.scatter(orthant_counts, avg_scores_km_rbf)
    plt.xlabel('Number of points in the orthant')
    plt.ylabel('Binary accuracy')
    plt.title('RBF SVM')
    plt.ylim(0., 1.)
    plt.savefig(plot_path_str+'acc_vs_counts_rbf.png')
    plt.figure(figsize=(8, 8))
    plt.scatter([ZERO_ORTHANT_INDEX], [avg_scores_km_rbf[ZERO_ORTHANT_INDEX]], c='g', s=25, label='Empty')
    plt.scatter(neighboring_orthants, avg_scores_km_rbf[neighboring_orthants], c='r', s=15, label='Neighbors')
    plt.scatter(non_neighboring_orthants, avg_scores_km_rbf[non_neighboring_orthants], c='b', s=5, label='Non-neighbors')
    plt.legend()
    plt.ylim(0., 1.)
    plt.xlabel('Orthant number')
    plt.ylabel('Binary accuracy')
    plt.title('RBF SVM')
    plt.savefig(plot_path_str+'acc_vs_orthant_rbf.png')

if args.pure:
    avg_scores_km_lin = np.mean(np.array(orthant_scores_km_lin), axis=0)
    plt.figure(figsize=(6, 6))
    plt.scatter(orthant_counts, avg_scores_km_lin)
    plt.xlabel('Number of points in the orthant')
    plt.ylabel('Binary accuracy')
    plt.title('Linear SVM')
    plt.ylim(0., 1.)
    plt.savefig(plot_path_str+'acc_vs_counts_lin.png')
    plt.figure(figsize=(8, 8))
    plt.scatter([ZERO_ORTHANT_INDEX], [avg_scores_km_lin[ZERO_ORTHANT_INDEX]], c='g', s=25, label='Empty')
    plt.scatter(neighboring_orthants, avg_scores_km_lin[neighboring_orthants], c='r', s=15, label='Neighbors')
    plt.scatter(non_neighboring_orthants, avg_scores_km_lin[non_neighboring_orthants], c='b', s=5, label='Non-neighbors')
    plt.legend()
    plt.ylim(0., 1.)
    plt.xlabel('Orthant number')
    plt.ylabel('Binary accuracy')
    plt.title('Linear SVM')
    plt.savefig(plot_path_str+'acc_vs_orthant_lin.png')

plt.figure(figsize=(20, 6))
for i in range(3):
    plt.subplot(1, 3, i+1)
    radial_probas_nn = np.mean(np.array(rad_probas_nn[i]), axis=0)
    plt.plot(radial_range, radial_probas_nn)
    plt.ylim(-0.05, 1.05)
    plt.title(key_orthants_types[i])
plt.suptitle('Average predicted probability of class 1 vs radius, NN')
plt.savefig(plot_path_str+'rad_dec_bound_nn.png')

plt.figure(figsize=(20, 6))
for i in range(3):
    plt.subplot(1, 3, i+1)
    radial_probas_km_ntk = np.mean(np.array(rad_probas_km_ntk[i]), axis=0)
    plt.plot(radial_range, radial_probas_km_ntk)
    plt.ylim(-0.05, 1.05)
    plt.title(key_orthants_types[i])
plt.suptitle('Average predicted probability of class 1 vs radius, NTK')
plt.savefig(plot_path_str+'rad_dec_bound_ntk.png')

if args.pure:
    plt.figure(figsize=(20, 6))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        radial_probas_km_rbf = np.mean(np.array(rad_probas_km_rbf[i]), axis=0)
        plt.plot(radial_range, radial_probas_km_rbf)
        plt.ylim(-0.05, 1.05)
        plt.title(key_orthants_types[i])
    plt.suptitle('Average predicted probability of class 1 vs radius, RBF')
    plt.savefig(plot_path_str+'rad_dec_bound_rbf.png')

if args.pure:
    plt.figure(figsize=(20, 6))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        radial_probas_km_lin = np.mean(np.array(rad_probas_km_lin[i]), axis=0)
        plt.plot(radial_range, radial_probas_km_lin)
        plt.ylim(-0.05, 1.05)
        plt.title(key_orthants_types[i])
    plt.suptitle('Average predicted probability of class 1 vs radius, Linear')
    plt.savefig(plot_path_str+'rad_dec_bound_lin.png')

add_log(f'Initial match fractions for class 0: {match_fracs_0_init}')
add_log(f'Final match fractions for class 0: {match_fracs_0_final}')
add_log(f'Initial match fractions for class 1: {match_fracs_1_init}')
add_log(f'Final match fractions for class 1: {match_fracs_1_final}')
