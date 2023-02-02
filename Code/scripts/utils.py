import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

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
    X, y, 
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

def plot_train_history(history):
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

    plt.show()

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
