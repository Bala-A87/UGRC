import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
import torch
from scripts.data.orthants import find_orthant, ORTHANTS, CENTRE, HIGH_COUNT, HIGH_SPREAD, generate_point
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from typing import Tuple, List, Iterable
from scripts.test import predict
from datetime import datetime
import os
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, labels) -> None:
        super().__init__()
        self.data = data
        self.labels = labels
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)

def make_dataloader(
    X: Iterable, 
    y: Iterable, 
    batch_size: int,
    shuffle: bool = False
) -> DataLoader:
    """
    Makes a torch DataLoader object from the given data.

    Args:
        X, y (iterables): Data and labels to be converted into DataLoader format.
        batch_size (int): batch size of the DataLoader.
        shuffle (bool, optional): whether to shuffle the data when creating the DataLoader.
            Defaults to False.
    
    Returns:
        torch.utils.data.DataLoader object containing len(X) (=len(y)) / batch_size batches of size batch_size from X and y.
    """
    return DataLoader(
        CustomDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle
    )

class EarlyStopping():
    def __init__(
        self,
        patience: int,
        min_delta: float,
        restore_best_weights: bool = False
    ) -> None:
        """
        Initializes an object to control overfitting in an NN by handling early stopping.

        Args:
            patience (int): Number of epochs to wait for improvement in validation loss
            min_delta (float): Minimum decrease in validation loss to count as an improvement
            restore_best_weights (bool, optional): Whether to restore the model to the weights at which it attained minimum
                validation loss when early stopping. Defaults to False.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_epoch = None
        self.best_loss = None
        if self.restore_best_weights:
            self.best_weights = None
    
    def check_stop(
        self,
        epoch: int,
        loss: float,
        weights: dict = None
    ):
        """
        Notifies the model training process on whether or not to stop training. Intended to be called during every epoch.

        Args:
            epoch (int): Current epoch number of training
            loss (float): Validation loss attained during current epoch
            weights (dict, optional): The state dict of the model during the current epoch.
                Defaults to None, but may cause unexpected behavior if not passed when restore_best_weights=True.
        
        Returns:
            False, if training is not to stop at this epoch.
            True, if training is to stop at this epoch, and restore_best_weights=False.
            The state dict of the model at which minimum loss is attained (dict), if training is to stop at this epoch,
                and restore_best_weights=True.
        """
        if (self.best_loss is None) or (self.best_loss - loss >= self.min_delta):
            self.best_epoch = epoch
            self.best_loss = loss
            if self.restore_best_weights:
                self.best_weights = weights
        elif epoch - self.best_epoch >= self.patience:
            if self.restore_best_weights:
                return self.best_weights
            else:
                return True
        return False

def plot_train_history(history, save_file: str = None):
    """
    Plots loss and metric score curves for training and validation using the history object returned by training function.
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.plot(history['epochs'], history['train_loss'], label='Training')
    plt.plot(history['epochs'], history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(history['epochs'], history['train_score'], label='Training')
    plt.plot(history['epochs'], history['val_score'], label='Validation')
    plt.title('Score')
    plt.legend()

    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)

def plot_ntk_corrs(preds_nn, preds_km_init, preds_km_inf):
    """
    Plots correlation between predictions of neural network and kernel machines with initial and final (T=0, T=inf) NTKs, with predictions of NN on x-axis and kernel machine predictions on y-axis, along with correlation (pearson's correlation coefficient) in the legend
    """
    corr_init, corr_inf = pearsonr(preds_nn.squeeze(), preds_km_init).statistic, pearsonr(preds_nn.squeeze(), preds_km_inf).statistic

    plt.figure(figsize=(8, 8))

    plt.scatter(preds_nn, preds_km_init, c='blue', s=6, label=f'T=0, corr: {corr_init:.3f}')
    plt.scatter(preds_nn, preds_km_inf, c='orange', s=6, label=f'T=inf, corr: {corr_inf:.3f}')
    plt.xlabel('NN preds')
    plt.ylabel('NTK SVM preds')
    plt.legend()

    plt.show()

from typing import Tuple

class Project7D_2D():
    """
    A class to construct a random 2D subspace in 7D space and find projections onto it.

    Initializes by randomly selecting two 7D vectors and orthonormalizing them, to act as an
    orthonormal basis for a random 2D subspace.
    """
    def __init__(self) -> None:
        self.v = torch.randn(7, 7)
        for i in range(7):
            for j in range(i):
                proj_ij = torch.dot(self.v[i], self.v[j])
                self.v[i] -= proj_ij * self.v[j]
            norm_i = torch.sqrt(torch.dot(self.v[i], self.v[i]))
            self.v[i] /= norm_i
    
    def fix_perp_comps(self, u: torch.Tensor) -> None:
        """
        Sets the components along basis vectors of 7D space besides the basis vectors of the 2D subspace
        to fixed values, to describe a 2D plane, using a given point u.

        Arg:
            u (torch.Tensor): Special vector in 7D space used to fix the 2D plane, a tensor of shape (7,)
        """
        u_projs = torch.tensor([torch.dot(u, self.v[i]) for i in range(7)])
        self.fixed_comps = u_projs[2:].reshape(1, -1)

    def project(self, u: torch.Tensor) -> torch.Tensor:
        """
        Projects a vector in 7D space into the randomly constructed 2D subspace and returns components along
        the orthonormal basis vectors.

        Arg:
            u (torch.Tensor): Initial vector in 7D space, a tensor of shape (7,)
        
        Returns:
            A tensor containing the scalar projections of u onto the random orthonormal basis of 7D space,
            the first two components of which indicate the "free" components, along the plane.
        """
        proj_1 = torch.dot(u, self.v[0]).reshape(1, 1)
        proj_2 = torch.dot(u, self.v[1]).reshape(1, 1)
        return torch.cat([proj_1, proj_2, self.fixed_comps], dim=1)

def plot_2d_visualization(
    X_train: torch.Tensor,
    Y_train: torch.Tensor, # maybe try improving this to only relevant data? not a necessity though ig
    models: List[torch.nn.Module],
    gif_save_file_name: str,
    centre: torch.Tensor = CENTRE,
    steps: Tuple[int, int] = (50, 50),
    orthant_min_count: int = 2*(HIGH_COUNT-HIGH_SPREAD),
    fps: int = 20,
    device: torch.device = 'cpu'
) -> None:
    """
    Performs a 2D planar visualization for 7D data, to witness evolution of classification boundary over training.

    Args:
        X_train (torch.Tensor): Training data
        Y_train (torch.Tensor): Training labels
        models (list): List of models, one corresponding to each epoch of training, to capture the state during each epoch
        gif_save_file_name (str): Name of file to save the animation as. Saved in the animations folder, under a directory with the time of saving as name. File name is expected to not contain the folder or the .gif extension
        centre (torch.Tensor, optional): Baseline point to fix the basis vectors not in the required plane.
            Defaults to tensor([4., 4., 4., 4., 4., 4., 4.]).
        steps (tuple, optional): Number of background/illustrative points to plot, expressed as number of samples along
        either direction of the 2 free basis vectors. Defaults to (50, 50).
        orthant_min_count (int, optional): Minimum number of points in an orthant required to choose the orthant randomly.
            Defaults to 34.
        fps (int, optional): fps rate for the gif. Defaults to 20.
        device (torch.device, optional): Device on which the model is trained. Defaults to cpu.
    """
    count = 0
    while count < orthant_min_count:
        random_orthant = int(torch.rand(1) * 127)
        train_orthants = torch.tensor([find_orthant(x) for x in X_train])
        X_train_random_orthant = X_train[train_orthants==random_orthant]
        Y_train_random_orthant = Y_train[train_orthants==random_orthant]
        count = len(Y_train_random_orthant)
    projector = Project7D_2D()
    projector.fix_perp_comps(ORTHANTS[random_orthant] * centre)
    X_train_random_orthant_proj = torch.cat([projector.project(x) for x in X_train_random_orthant])
    fig, ax = plt.subplots()
    plt.scatter(X_train_random_orthant_proj[:, 0], X_train_random_orthant_proj[:, 1], s=20, c=Y_train_random_orthant, cmap='PiYG')
    xlim = plt.xlim()
    ylim = plt.ylim()
    X_grid, Y_grid = torch.meshgrid([
        torch.linspace(xlim[0], xlim[1], steps[0]),
        torch.linspace(ylim[0], ylim[1], steps[1])
    ], indexing='xy')
    X_all, Y_all = X_grid.reshape(-1, 1), Y_grid.reshape(-1, 1)
    X_input = torch.matmul(X_all, projector.v[0].unsqueeze(0)) + torch.matmul(Y_all, projector.v[1].unsqueeze(0)) + torch.matmul(projector.fixed_comps, projector.v[2:])

    def animate(i):
        model = models[i]
        ax.clear()
        ax.scatter(X_train_random_orthant_proj[:, 0], X_train_random_orthant_proj[:, 1], s=20, c=Y_train_random_orthant, cmap='PiYG')
        Z_all = torch.round(predict(model, X_input, device)) 
        sc = ax.scatter(X_all, Y_all, s=8, c=Z_all, cmap='PiYG', alpha=0.25)
        ax.set_title('Epoch '+str(i+1))
        fig.tight_layout()
        return sc
    
    ani = FuncAnimation(fig=fig, func=animate, frames=range(len(models)), interval=50, repeat=True)
    writer = PillowWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

    now = datetime.now()
    folder_name = 'animations/' + now.strftime("%d%m%Y_%H%M")
    os.mkdir(folder_name)

    ani.save(folder_name+'/'+gif_save_file_name+'.gif', writer=writer)

def plot_radial_visualization(
    models: List[torch.nn.Module],
    mp4_save_file_name: str,
    orthant_counts: torch.Tensor,
    main_orthant: int = None,
    num_samples_per_orthant: int = 100,
    range_start: float = 0.5,
    range_stop: float = 2.6,
    range_step: float = 0.1,
    centre: torch.Tensor = CENTRE,
    nrows: int = 2,
    ncols: int = 4,
    fps: int = 20,
    device: torch.device = 'cpu'
) -> None:
    """
    Plots a visualization of the variation of the probability of class 1 with radius of the sphere, in a specified/random 
    orthant and its surrounding 7 neighbors, evolving with training.

    Args:
        models (list): List of models, one corresponding to each epoch of training, to capture the state during each epoch
        mp4_save_file_name (str): Name of the file to save the generated animation in, as an mp4 file. Saved in 
        animations folder under a subfolder corresponding to the time of video file creation. Only the file name is
        expected, without any directories or extension.
        orthant_counts (torch.Tensor): Number of data points used for training in each orthant.
            Tensor of shape (128,).
        main_orthant (int, optional): The main orthant around which to visualize.
            Defaults to None, in which case a random orthant is chosen.
        num_samples_per_orthant (int, optional): The number of data points to use to compute the average predicted
            probability for each orthant. Defaults to 100.
        range_start (float, optional): The lower bound (inclusive) for the radii used to plot the variation on. 
            Defaults to 0.5. Must be > 0 to be meaningful.
        range_stop (float, optional): The upper bound (exclusive) for the radii used to plot the variation on. 
            Defaults to 2.6. Must be > 0 to be meaningful.
        range_step (float, optional): The step size for each increment in radius. Defaults to 0.1.
            Must be > 0 to be meaningful.
        centre (torch.Tensor, optional): The centre of the sphere (signless).
            Defaults to tensor([4., 4., 4., 4., 4., 4., 4.]).
        nrows (int, optional): The number of rows in the subplots produced. Defaults to 2.
        ncols (int, optional): The number of columns in the subplots produced. Defaults to 4.
            nrows * ncols being 8 allows for maximum subplot usage, since 8 subplots are produced.
        fps (int, optional): The frame rate of the produced animation. Defaults to 20.
        device (torch.device, optional): The device on which the model is used to predict on.
            Defaults to cpu.
    """
    if main_orthant is None:
        main_orthant = np.random.choice(range(128), 1)
    reqd_orthants = ORTHANTS[main_orthant].reshape(1, -1)
    for i in range(7):
        new_orthant = torch.clone(ORTHANTS[main_orthant]).reshape(-1,)
        new_orthant[i] *= -1.
        reqd_orthants = torch.cat([reqd_orthants, new_orthant.reshape(1, -1)])
    X_radial = torch.cat([
        torch.cat([
            torch.cat([
                generate_point(radius, centre, reqd_orthants[i]).reshape(1, -1) for j in range(num_samples_per_orthant)
            ]).reshape(1, num_samples_per_orthant, 7) for radius in np.arange(range_start, range_stop, range_step)
        ]).reshape(1, int((range_stop-range_start)/range_step), num_samples_per_orthant, 7) for i in range(8)
    ])

    fig, ax = plt.subplots(nrows, ncols, figsize=(18, 9))

    def animate(i):
        Y_radial = torch.cat([
            torch.tensor([
                torch.mean(predict(models[i], X_radial[j][r], device)) for r in range(int((range_stop-range_start)/range_step))
            ]).reshape(1, -1) for j in range(8)
        ])
        for i1 in range(nrows):
            for i2 in range(ncols):
                ax[i1][i2].clear()
                ax[i1][i2].plot(torch.arange(range_start, range_stop, range_step), Y_radial[int(ncols*i1 + i2)])
                ax[i1][i2].set_title('Orthant '+str(find_orthant(reqd_orthants[int(ncols*i1 + i2)]))+', '+str(orthant_counts[find_orthant(reqd_orthants[int(ncols*i1 + i2)])])+' points')
                ax[i1][i2].annotate(str(Y_radial[int(ncols*i1 + i2)][int((1.-range_start)/range_step)]), (1., Y_radial[int(ncols*i1 + i2)][int((1.-range_start)/range_step)]))
                ax[i1][i2].annotate(str(Y_radial[int(ncols*i1 + i2)][int((2.-range_start)/range_step)]), (2., Y_radial[int(ncols*i1 + i2)][int((2.-range_start)/range_step)]))
                ax[i1][i2].set_ylim(-0.1, 1.1)
        plt.suptitle('Epoch '+str(i)+', avg probability of class 1 vs radius')
    
    ani = FuncAnimation(fig, animate, frames=len(models), interval=1000/fps, repeat=True)
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

    now = datetime.now()
    folder_name = 'animations/' + now.strftime("%d%m%Y_%H%M")
    os.mkdir(folder_name)

    ani.save(folder_name+'/'+mp4_save_file_name+'.mp4', writer=writer)
