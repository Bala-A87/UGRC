from torch import nn

class SimpleNN(nn.Module):
    """
    Subclasses torch.nn.Module to implement a simple multi-purpose neural network.

    Args:
        input_size (int): Number of input features.
        output_size (int, optional): Number of outputs to produce. Defaults to 1.
        task_type (str, optional): Type of task to perform.
            Allowed values:
                'regression',
                'classification'
            Defaults to 'classification'.
        The following combinations are possible:
            output_size: 1, task_type: 'regression' => Classic regression
            output_size: 1, task_type: 'classification' => Binary classification
            output_size: >2, task_type: 'classification' => Multiclass classification
        hidden_layers (int, optional): Number of hidden layers in the model. Defaults to 2.
        hidden_units (int, optional): Number of neurons per hidden layer in the model. Defaults to 32.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        task_type: str = 'classification',
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
            
        if task_type.lower() == 'regression':
            self.output = nn.Sequential(
                nn.Linear(in_features=hidden_units, out_features=output_size)
            )
        elif task_type.lower() == 'classification':
            if output_size == 1:
                self.output = nn.Sequential(
                    nn.Linear(in_features=hidden_units, out_features=output_size),
                    nn.Sigmoid()
                )
            else:
                self.output = nn.Sequential(
                    nn.Linear(in_features=hidden_units, out_features=output_size),
                    nn.Softmax(dim=1)
                )
    
    def forward(self, x):
        return self.output(self.hidden(self.input(x)))
