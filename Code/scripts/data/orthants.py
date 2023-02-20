import torch
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from numpy.random import choice

SIGNS = [-1., 1.] 

ORTHANTS = torch.stack(
    [torch.tensor([s1, s2, s3, s4, s5, s6, s7])
    for s1 in SIGNS
    for s2 in SIGNS
    for s3 in SIGNS
    for s4 in SIGNS
    for s5 in SIGNS
    for s6 in SIGNS
    for s7 in SIGNS])   # Representing each orthant as the sign of each of its 7 coordinates/wrt each of its 7 axes

LOW_COUNT = 4   # Average no. of points in a "rare" orthant
HIGH_COUNT = 25 # Average no. of points in a "typical" orthant
LOW_SPREAD = 2  # Allowed unidirectional deviation for no. of points in a "rare" orthant (i.e., 2 to 6 points)
HIGH_SPREAD = 8 # Allowed unidirectional deviation for no. of points in a "typical" orthant (i.e., 17 to 33 points)
TEST_COUNT = 64 # Test data points per orthant, for final pattern
LOW_FRAC = 0.25 # Fraction of "rare" orthants
ZERO_FRAC = 0.25    # Fraction of empty orthants from rare orthants

CENTRE = 4 * torch.ones(7)

LOW_RADIUS = 1.
HIGH_RADIUS = 2.

CHOICES = [0., 1.]

def find_orthant(point: torch.Tensor) -> int:
    """
    Given a point, determines the orthant it lies in as an index of ORTHANTS

    Args:
        point (torch.Tensor): A random point in 7d space, a tensor of shape (7,)
    
    Returns:
        orthant_no (int): The index of ORTHANTS corresponding to the orthant point lies in
            i.e., orthant_no satisfies ORTHANTS[orthant_no] == torch.sign(point)
    
    Example usage:
    >>> find_orthant(tensor([1., 3., -2., -4., 3., 2.2, -9.]))
    102
    """
    orthant_no = 0
    for dim in point:
        orthant_no *= 2
        if dim > 0:
            orthant_no += 1
    return int(orthant_no)

def generate_point(
    radius: float,
    centre: torch.Tensor,
    orthant: torch.Tensor
) -> torch.Tensor:
    """
    Generates a point on a 7-sphere of centre orthant * centre with radius radius, where
    orthant denotes the signs wrt all 7 axes

    Args:
        radius (float): radius of the 7-sphere
        centre (float): unsigned centre of the 7-sphere, i.e., without orthant-specificity
        orthant (torch.Tensor): signs wrt the 7 axes for the resulting point, of shape (7,)
    
    Returns:
        x (torch.Tensor): a point on the specified sphere, of shape (7,)
    
    Example usage:
    >>> generate_point(2, tensor([4., 4., 4., 4., 4., 4., 4.], tensor([1., 1., 1., 1., 1., 1., 1.]))
    tensor([2., 4., 4., 4., 4., 4., 4.])
    """
    phi = torch.rand((6,)) * torch.pi
    phi[5] *= 2
    diff_unit = torch.tensor([
        torch.cos(phi[0]),
        torch.sin(phi[0])*torch.cos(phi[1]),
        torch.sin(phi[0])*torch.sin(phi[1])*torch.cos(phi[2]),
        torch.sin(phi[0])*torch.sin(phi[1])*torch.sin(phi[2])*torch.cos(phi[3]),
        torch.sin(phi[0])*torch.sin(phi[1])*torch.sin(phi[2])*torch.sin(phi[3])*torch.cos(phi[4]),
        torch.sin(phi[0])*torch.sin(phi[1])*torch.sin(phi[2])*torch.sin(phi[3])*torch.sin(phi[4])*torch.cos(phi[5]),
        torch.sin(phi[0])*torch.sin(phi[1])*torch.sin(phi[2])*torch.sin(phi[3])*torch.sin(phi[4])*torch.sin(phi[5])
    ])
    return torch.mul(centre + radius * diff_unit, orthant)

def generate_flips(flip_proba: float = 0.):
    flips = torch.zeros(128)
    if flip_proba > 0.:
        probas = [1 - flip_proba, flip_proba]
        flips = torch.tensor(choice(a=CHOICES, size=128, p=probas))
    return flips
    
def generate_train_data(
    low_count: int = LOW_COUNT,
    high_count: int = HIGH_COUNT,
    low_spread: int = LOW_SPREAD,
    high_spread: int = HIGH_SPREAD,
    low_frac: float = LOW_FRAC,
    zero_frac: float = ZERO_FRAC,
    centre: torch.Tensor = CENTRE,
    low_radius: float = LOW_RADIUS,
    high_radius: float = HIGH_RADIUS,
    flips: torch.Tensor = torch.zeros(128)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates train data of concentric 7-spheres in each orthant of 7d space. The 7-spheres are centred at
    centre (with signs adjusted as per the orthant), with radii low_radius (labelled 0) and high_radius
    (labelled 1). Some orthants are populated sparsely (number of points of each class in [low_count-low_spread,
    low_count+low_spread]), some more densely (number of points of each class in [high_count-high_spread,
    high_count+high_spread]), and some not at all. 1-low_frac fraction of orthants is populated densely, 
    low_frac*(1-zero_frac) fraction is populated sparsely, and low_frac*zero_frac fraction is not populated. 
    In addition to this, some orthants have the class labels flipped given by flips.

    Args:
        low_count (int, optional): Average number of data points of each class in a sparsely populated orthant.
            Defaults to 4.
        high_count (int, optional): Average number of data points of each class in a densely populated orthant.
            Defaults to 25.
        low_spread (int, optional): Maximum permissible deviation for number of points in a sparsely
            populated orthant. Defaults to 2.
        low_spread (int, optional): Maximum permissible deviation for number of points in a densely
            populated orthant. Defaults to 8.
        low_frac (float, optional): Fraction of orthants sparsely/not at all populated.
            Defaults to 0.25.
        zero_frac (float, optional): Fraction of sparsely populated orthants to not populate.
            Defaults to 0.25.
        centre (torch.Tensor, optional): Signless centre for data in each orthant.
            Defaults to tensor([4., 4., 4., 4., 4., 4., 4.]).
        low_radius (float, optional): Radius of inner spheres. Defaults to 1.
        high_radius (float, optional): Radius of outer spheres. Defaults to 2.
        flips (torch.Tensor, optional): Whether label is flipped in each orthant (1 if so, else 0).
            Defaults to torch.zeros(128) (no flips).
        
    Returns:
        A 3-tuple of torch.Tensors, (X_training, Y_training, orthant_counts)
        X_training: torch.Tensor of shape (num_samples, 7), the data points
        Y_training: torch.Tensor of shape (num_samples, 1), the labels
        orthant_counts: torch.Tensor of shape (128,), the number of data points in each orthant
    """
    high_orthant_indices, lower_orthant_indices = train_test_split(np.arange(128), test_size=low_frac)
    low_orthant_indices, zero_orthant_indices = train_test_split(lower_orthant_indices, test_size=zero_frac)

    X_training_low_0 = torch.cat([
        torch.cat([
            generate_point(low_radius, centre, orthant).reshape(1, -1) for i in range(int((2*torch.rand(1) - 1) * low_spread + low_count))
        ]) for orthant in ORTHANTS[low_orthant_indices]
    ])
    X_training_low_1 = torch.cat([
        torch.cat([
            generate_point(high_radius, centre, orthant).reshape(1, -1) for i in range(int((2*torch.rand(1) - 1) * low_spread + low_count))
        ]) for orthant in ORTHANTS[low_orthant_indices]
    ])
    X_training_high_0 = torch.cat([
        torch.cat([
            generate_point(low_radius, centre, orthant).reshape(1, -1) for i in range(int((2*torch.rand(1) - 1) * high_spread + high_count))
        ]) for orthant in ORTHANTS[high_orthant_indices]
    ])
    X_training_high_1 = torch.cat([
        torch.cat([
            generate_point(high_radius, centre, orthant).reshape(1, -1) for i in range(int((2*torch.rand(1) - 1) * high_spread + high_count))
        ]) for orthant in ORTHANTS[high_orthant_indices]
    ])

    Y_training_low_0 = torch.zeros(len(X_training_low_0), 1)
    Y_training_low_1 = torch.ones(len(X_training_low_1), 1)
    Y_training_high_0 = torch.zeros(len(X_training_high_0), 1)
    Y_training_high_1 = torch.ones(len(X_training_high_1), 1)

    X_training = torch.cat([
        X_training_low_0,
        X_training_low_1,
        X_training_high_0,
        X_training_high_1
    ])
    Y_training = torch.cat([
        Y_training_low_0,
        Y_training_low_1,
        Y_training_high_0,
        Y_training_high_1
    ])
    orthant_counts = torch.zeros(128)
    for x in X_training:
        orthant_counts[find_orthant(x)] += 1
    
    Y_training = torch.tensor([Y_training[i] if flips[find_orthant(X_training[i])] == 0 else 1-Y_training[i] for i in range(len(Y_training))]).reshape(-1, 1)

    return X_training, Y_training, orthant_counts

def generate_test_data(
    test_count: int = TEST_COUNT,
    centre: torch.Tensor = CENTRE,
    low_radius: float = LOW_RADIUS,
    high_radius: float = HIGH_RADIUS,
    flips: torch.Tensor = torch.zeros(128)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates test data of 2 concentric 7-spheres in each orthant of 7d space. The spheres have an unsigned
    centre of centre, the inner sphere has radius low_radius (class label 0), the outer high_radius (label 1).
    test_count number of data points are generated for each class in each orthant. Additionally, some orthants
    have the labels of its data points flipped, as given by flips.

    Args:
        test_count (int, optional): Number of data points of each class in each orthant.
            Defaults to 64.
        centre (torch.Tensor, optional): Signless centre for data in each orthant.
            Defaults to tensor([4., 4., 4., 4., 4., 4., 4.]).
        low_radius (float, optional): Radius of inner spheres. Defaults to 1.
        high_radius (float, optional): Radius of outer spheres. Defaults to 2.
        flips (torch.Tensor, optional): Whether label is flipped in each orthant (1 if so, else 0).
            Defaults to torch.zeros(128) (no flips).
    
    Returns:
        A 2-tuple of torch.Tensors, (X_test, Y_test, flips)
        X_test: torch.Tensor of shape (num_samples, 7), the data points
        Y_test: torch.Tensor of shape (num_samples, 1), the labels
    """
    X_test = torch.cat([
        torch.cat([
            torch.cat([
                generate_point(low_radius, centre, orthant).reshape(1, -1) for i in range(test_count)
            ]),
            torch.cat([
                generate_point(high_radius, centre, orthant).reshape(1, -1) for i in range(test_count)
            ])
        ]).reshape(1, 2*test_count, 7) for orthant in ORTHANTS
    ])
    Y_test = torch.cat([
        torch.cat([
            torch.zeros(test_count, 1),
            torch.ones(test_count, 1)
        ]).reshape(1, 2*test_count, 1) for orthant in ORTHANTS
    ])

    Y_test = torch.cat([(Y_test[i] if flips[i] == 0 else 1-Y_test[i]).reshape(1, 2*test_count, 1) for i in range(len(Y_test))])

    return X_test, Y_test
