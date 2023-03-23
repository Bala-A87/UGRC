from torch import nn

class SimpleNN(nn.Module):
    """
    Subclasses torch.nn.Module to implement a simple multi-purpose neural network.

    Args:
        input_size (int): Number of input features.
        output_size (int, optional): Number of outputs to produce. Defaults to 1.
        activation: Activation function (class, not object) to use for the non-output layers.
            Defaults to torch.nn.ReLU.
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
        activation = nn.ReLU,
        task_type: str = 'classification',
        hidden_layers: int = 2,
        hidden_units: int = 32
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.task_type = task_type
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.input = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_units),
            activation()
        )
        self.hidden = nn.Sequential()
        for i in range(hidden_layers):
            self.hidden.append(nn.Linear(in_features=hidden_units, out_features=hidden_units))
            self.hidden.append(activation())
            
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

    def clone(self) -> nn.Module:
        """
        Returns a clone of the current SimpleNN object.
        """
        new_model = SimpleNN(
            input_size=self.input_size,
            output_size=self.output_size,
            activation=self.activation,
            task_type=self.task_type,
            hidden_layers=self.hidden_layers,
            hidden_units=self.hidden_units
        )
        new_model.load_state_dict(self.state_dict())
        return new_model
    
    def freeze_first(self) -> None:
        """
        Freezes the weights of all the layers of the model except the last layer.
        """
        for param in self.input.parameters():
            param.requires_grad = False
            print('Input layer weights frozen')
        
    def freeze_last(self) -> None:
        """
        Freezes the weights of the last layer of the model.
        """
        for param in self.hidden.parameters():
            param.requires_grad = False
            print('Hidden layer weights frozen')
        for param in self.output.parameters():
            param.requires_grad = False
            print('Output layer weights frozen')
    
    def revive_last(self) -> None:
        """
        Revives (allows learning) the weights of the last layer of the model.
        """
        for param in self.hidden.parameters():
            param.requires_grad = True
            print('Hidden layer weights revived')
        for param in self.output.parameters():
            param.requires_grad = True
            print('Output layer weights revived')
    
    def reinit_last(self) -> None:
        """
        Reinitializes the weights of the last layer of the model.
        """
        for child in self.hidden.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()
                print('Hidden layer weights reinitialized')
        for child in self.output.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()
                print('Output layer weights reinitialized')
