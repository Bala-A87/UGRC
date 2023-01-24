import torch

def expNegMod(x):
    """
    For input x, returns exp(-abs(x))
    """
    return torch.exp(-torch.abs(x))

class ExpNegMod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return expNegMod(x)

def expNegSq(x):
    """
    For input x, returns exp(-x**2)
    """
    return torch.exp(-x**2)

class ExpNegSq(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return expNegSq(x)

def discExpNegMod(x):
    """
    For input x, returns relu(1 - 0.5*relu(abs(x)))
    """
    relu = torch.nn.ReLU()
    return relu(1 - 0.5*relu(torch.abs(x)))

class DiscExpNegMod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return discExpNegMod(x)

def mod(x):
    return torch.abs(x)

class Mod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return mod(x)

def sq(x):
    return x**2

class Sq(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return sq(x)

def sinoverx(x):
    return torch.sin(x)/x

class SinOverx(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return sinoverx(x)
    
def oneNegMod(x):
    return 1-torch.abs(x)

class OneNegMod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return oneNegMod(x)         
