import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

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
