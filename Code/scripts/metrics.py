import torch

class BinaryAccuracy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, preds, y):
        preds_class = torch.round(preds)
        return torch.sum(preds_class == y)/len(y)
