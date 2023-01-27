import torch

def predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device = 'cpu'
) -> torch.Tensor:
    """
    Computes the predictions of the give model on given data.

    Args:
        model (torch.nn.Module): The (trained) model to predict with
        x (torch.Tensor): The data to predict on
        device (torch.device, optional): The device to use for prediction. 
            Defaults to cpu.
    
    Returns:
        y (torch.Tensor): The predictions of model on x
    """
    model.eval()
    with torch.inference_mode():
        return model(x.to(device))
