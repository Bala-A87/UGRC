import torch
from typing import Dict, List
from scripts.utils import EarlyStopping

def train_classifier(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metric: torch.nn.Module,
    epochs: int,
    early_stopping: EarlyStopping = None,
    verbose: int = 2,
    device: torch.device = 'cpu'
) -> Dict[str, List[float]]:
    """
    Trains a torch NN classifier (binary) with the provided arguments.

    Args:
        model (torch.nn.Module): The model to train.
        train_dataloader (torch.utils.data.DataLoader): Train data in DataLoader form.
        val_dataloader (torch.utils.data.DataLoader): Validation data in DataLoader form.
        loss_fn (torch.nn.Module): Loss function to use for training.
        optimizer (torch.optim.Optimizer): Optimizer to use for weights updation.
        metric (torch.nn.Module): Evaluation metric.
        epochs (int): Number of epochs to train for.
        early_stopping (EarlyStopping, optional): EarlyStopping object for regularization.
            Defaults to None.
        verbose (int, optional): Verbosity of information printed during training.
            < 1: No messages printed.
            >= 1, < 2: Beginning and ending messages printed.
            >= 2: Losses and scores are also printed for each epoch of training.
            Defaults to 2.
        device (torch.device, optional): Device to train the model on.
            Defaults to cpu.
    
    Returns:
        A history object containing all the training information, in the form of a dict with str keys and values of type list of floats.

        Valid keys of the dict: 
            epochs
            train_loss
            train_score
            val_loss
            val_score
    """
    if verbose >= 1:
        print(f"Training model for {epochs} epochs. Early stopping{' not' if early_stopping is None else ''} enabled.")
    history = {
        'epochs': [],
        'train_loss': [],
        'train_score': [],
        'val_loss': [],
        'val_score': []
    }
    for epoch in range(epochs):
        train_loss, train_score = 0.0, 0.0
        for (X, y) in train_dataloader:
            X, y = X.to(device), y.to(device)
            
            model.train()
            preds_prob = model(X)
            loss = loss_fn(preds_prob, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            train_score += metric(preds_prob, y)
        
        train_loss /= len(train_dataloader)
        train_score /= len(train_dataloader)

        model.eval()
        with torch.inference_mode():
            val_loss, val_score = 0.0, 0.0
            for (X, y) in val_dataloader:
                X, y = X.to(device), y.to(device)

                preds_prob = model(X)
                val_loss += loss_fn(preds_prob, y)
                val_score += metric(preds_prob, y)
        
            val_loss /= len(val_dataloader)
            val_score /= len(val_dataloader)

        history['epochs'].append(epoch+1)
        history['train_loss'].append(train_loss.item())
        history['train_score'].append(train_score)
        history['val_loss'].append(val_loss.item())
        history['val_score'].append(val_score)

        if verbose >= 2:
            print(f'Epoch: {epoch+1} => Train loss: {train_loss:.6f}, Train score: {train_score:.6f}, Val loss: {val_loss:.6f}, Val score: {val_score:.6f}')
        
        if early_stopping is not None:
            stop = early_stopping.check_stop(epoch=epoch+1, loss=val_loss, weights=model.state_dict())
            if stop:
                if not(stop == True):
                    model.load_state_dict(stop)
                if verbose >= 1:
                    print('Training stopped.')
                return history
    if verbose >= 1:
        print('Training complete.')
    return history
