from torch import nn

class SimpleNN(nn.Module):
    """
    Subclasses torch.nn.Module to implement a simple binary classification neural network.

    Args:
        input_size (int): Number of input features.
        hidden_layers (int, optional): Number of hidden layers in the model. Defaults to 2.
        hidden_units (int, optional): Number of neurons per hidden layer in the model. Defaults to 32.
    """
    def __init__(
        self,
        input_size: int,
        hidden_layers: int = 2,
        hidden_units: int = 32
    ) -> None:
        super().__init__()
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.input = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_units),
            nn.ReLU()
        )
        self.hidden = nn.Sequential()
        for i in range(hidden_layers):
            self.hidden.append(nn.Linear(in_features=hidden_units, out_features=hidden_units))
            self.hidden.append(nn.ReLU())
        self.output = nn.Sequential(
            nn.Linear(in_features=hidden_units, out_features=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.output(self.hidden(self.input(x)))
