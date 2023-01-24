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
        return torch.tensor(torch.sum(preds_class == y)/len(y))
