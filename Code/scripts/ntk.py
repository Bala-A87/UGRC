import torch

def get_grads(u, model: torch.nn.Module) -> torch.Tensor:
    """
    Computes gradient of output of model for input u (single input) wrt parameters of model

    Args:
        u: Input data row (1 entry) of shape (num_features,)
        model (torch.nn.Module): A neural network model that can accept u as input 
    
    Returns:
        grad (torch.Tensor): Gradient of model(u) wrt model.parameters() of size (1, num_params) where num_params = number of parameters in model
    """
    return torch.cat([torch.reshape(grads, (-1,)) for grads in list(torch.autograd.grad(model(u), model.parameters()))]).reshape(1, -1)

def compute_ntk(U, V, model: torch.nn.Module) -> torch.Tensor:
    """
    Computes NTK for data matrices U, V with neural network model

    Args:
        U: Data matrix 1 of shape (num_samples_U, num_features)
        V: Data matrix 2 of shape (num_samples_V, num_features)
        model (torch.nn.Module): A neural network model that can accept U and V as inputs

    Returns:
        K (torch.Tensor): NTK of U & V wrt model, of shape (num_samples_U, num_samples_V) 
    """
    phi_u = torch.cat([get_grads(u, model) for u in U])
    phi_v = torch.cat([get_grads(v, model) for v in V])
    return torch.matmul(phi_u, phi_v.T)

class NTK():
    """
    Class for NTK computation, designed to be used along with sklearn.svm.SVR/SVC

    Args:
        model (torch.nn.Module): Neural network model to be used for NTK computation.
    """
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
    
    def get_ntk(self, U, V) -> torch.Tensor:
        """
        Wrapper function for compute_ntk with model provided by the class, meant to be provided as callable kernel for sklearn.svm.SVR/SVC

        Args:
            U, V: Data matrices of shape (num_samples_U, num_features), (num_samples_V, num_features) compatible with model
        
        Returns:
            NTK of U and V wrt self.model, i.e., returns result of compute_ntk(U, V, self.model
        """
        return compute_ntk(U, V, self.model)
