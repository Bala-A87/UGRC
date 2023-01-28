import torch

class BinaryAccuracy(torch.nn.Module):
    """
    Computes accuracy of predictions for a binary classification problem.

    Created object must be called with two arguments:
        preds: Predicted probabilities of positive (1)
        y: True class labels (must be 0/1)
    """

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, preds, y):
        preds_class = torch.round(preds)
        return torch.sum(preds_class == y)/len(y)

class NegMeanSquaredError(torch.nn.Module):
    """
    Computes negative mean squared error for a regression problem.

    Created object must be called with two arguments:
        preds: Predicted values
        y: True values
    """

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, preds, y):
        return -torch.sum((preds.detach() - y)**2)/len(y)

class CategoricalAccuracy(torch.nn.Module):
    """
    Computes accuracy of predictions for a categorical (multi-class) classification problem.

    Created object must be called with two arguments:
        preds: Predictions in a one-hot encoded format, i.e., shape (num_samples, num_classes)
        y: True class labels with shape (num_samples, ) with each entry in range(num_classes) 
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, preds, y):
        preds_class = torch.argmax(preds, dim=1)
        return torch.sum(preds_class == y)/len(y)
