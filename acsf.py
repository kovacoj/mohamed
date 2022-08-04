'''Atom-centered symmetry functions.
Reference:
    - [Jörg Behler "Atom-centered symmetry functions for constructing high-dimensional neural network potentials", J. Chem. Phys. 134, 074106 (2011)](https://doi.org/10.1063/1.3553717) 
'''

# TODO: prevent theta from being in gradient?
# TODO: Add weighted-acsfs
# FIXME: angular ACSFs are erronious


import torch
from utils import identity


def cutoff_cos(r, rC=10):
    """Cosine cutoff function.

    The cosine cutoff function with parameter `rC`, cutoff radius,
    at which the function goes to zero.

    Args:
        r: Input tensor. Radial distances.
        rC: A scalar. Cutoff radius.

    Returns:
        Pairwise, `cos(pi*r/rC)+1)/2` for `r <= rC`, otherwise `0`.
    """
    return torch.where(r > rC, torch.zeros(r.shape),
                (torch.cos(torch.pi*r/rC)+1)/2)

def cutoff_tanh(r, rC=10):
    """Tanh cutoff function
    
    Continuous first and second derivatives,
    that go to zero at the cutoff radius `rC`.

    Returns:
        Pairwise, `tanh(1-r/rC)**3)` for `r <= rC`, otherwise `0`.
    """
    return torch.where(r > rC, torch.zeros(r.shape),
                torch.tanh(1-r/rC)**3)

def cutoff_exp(r, rC=10):
    """Exponential cutoff function
    
    Continuous derivatives up to infinite order at the cutoff radius.
    
    Returns:
        Pairwise, `exp(1-1/(1-(r/rC)**2)))` for `r <= rC`, otherwise `0`.
    """
    return torch.where(r > rC, torch.zeros(r.shape),
                torch.exp(1 - 1 / (1 - (r / rC)**2)))

def G1(r, rC=10, cutoff=cutoff_cos):
    """
    Sums over `dim = 1` as first dim is reserved for batch.

    Args:
        r: Input tensor. Radial distances.
        rC: A scalar. Cutoff radius.
        cutoff: Pytorch function. Cutoff function.

    Returns:
        `cutoff(r, rC).sum(dim = 1)`
    """
    return cutoff(r, rC).sum(dim=1, keepdim=True)

def G2(r, eta=1, rS=0, rC=10, cutoff=cutoff_cos):
    """
    Returns:
        `sum(exp(-eta * (r - rS)**2) * cutoff(r, rC), dim=1)`
    """
    return (torch.exp(-eta * (r - rS)**2) * cutoff(r, rC)).sum(dim=1, keepdim=True)

def G3(r, kappa, rC=10, cutoff=cutoff_cos):
    """
    Returns:
        `sum(cos(kappa * r) * cutoff(r, rC), dim = 1)`
    """
    return (torch.cos(kappa * r) * cutoff(r, rC)).sum(dim=1, keepdim=True)

def S1(r, alpha=1, rC=10, cutoff=cutoff_cos):
    """
    Returns:
        `sum(fC(r) / r**alpha, dim = 1)`
    """
    return (cutoff(r, rC) / r**alpha).sum(dim=1, keepdim=True)

def S2(r, eta, rS, rC=10, cutoff=cutoff_cos):
    """
    Returns:
        `sum(exp(-eta*abs(r - rS))*fC(r), dim = 1)`
    """
    return (torch.exp(-eta * torch.abs(r - rS)) * cutoff(r, rC)).sum(dim = 1, keepdim=True)

def angular(theta, lambd, zeta):
    """Angular part of angular-ACSFs for trimer.
    TODO: find better name
    Returns:
        `2**(1 - zeta) * (1 + lambd * cos(theta))**zeta`
    """
    return 2**(1-zeta) * (1+lambd*torch.cos(theta))**zeta

def radial(r, eta, rC, cutoff=cutoff_cos):

    return (torch.exp(-eta*(r**2))*cutoff(r**2, rC)).prod(dim=1, keepdim=True)

def G4_trimer(r, zeta=1, lambd=1, eta=1, rC=10, cutoff=cutoff_cos):
    """
    TODO: add docstring
    Returns:
        `2**(1 - zeta) * (1 + lambd * cos(theta))**zeta * exp(-eta*r**2).prod(dim=1)`
    """
    theta = torch.arccos((r[:, 0]**2+r[:, 1]**2-r[:, 2]**2)/(2*r[:, 0]*r[:, 1])).view(-1, 1)
    angular_part = angular(theta=theta, lambd=lambd, zeta=zeta)
    radial_part = radial(r=r, eta=eta, rC=rC, cutoff=cutoff)

    return angular_part*radial_part

def S4_trimer(r, zeta=1, lambd=1, eta=1, rC=10, cutoff=cutoff_cos):
    """
    Returns:
        `2**(1 - zeta) * (1 + lambd * cos(theta))**zeta * exp(-eta*abs(r)).prod(dim=1)`
    """
    theta = torch.arccos((r[:, 0]**2+r[:, 1]**2-r[:, 2]**2)/(2*r[:, 0]*r[:, 1])).view(-1, 1)
    angular_part = angular(theta=theta, zeta=zeta, lambd=lambd)
    radial_part = (torch.exp(-eta*r)*cutoff(r, rC)).prod(dim=1, keepdim=True)

    return angular_part*radial_part

def G5_trimer(r, zeta=1, lambd=1, eta=1, rC=10, cutoff=cutoff_cos):
    """Same as G4_trimer but without the 'jk' term.

    Returns:
        `2**(1 - zeta) * (1 + lambd * cos(theta))**zeta * exp(-eta*r**2).prod(dim=1)`
    """
    theta = torch.arccos((r[:, 0]**2+r[:, 1]**2-r[:, 2]**2)/(2*r[:, 0]*r[:, 1])).view(-1, 1)
    angular_part = angular(theta=theta, lambd=lambd, zeta=zeta)
    radial_part = radial(r=r[:, :-1], eta=eta, rC=rC, cutoff=cutoff)

    return angular_part*radial_part

def W1(r, Z, rC=10, cutoff=cutoff_cos, g=identity):
    """Weighted radial ACSF for G1
    Returns:
        `sum(g(Z) * cutoff(r, rC), dim = 1)`
    """
    return (g(Z)*cutoff(r, rC)).sum(dim=1, keepdim=True)

def W2(r, Z, eta=1, rS=0, rC=10, cutoff=cutoff_cos, g=identity):
    """Weighted radial ACSF for G2
    Returns:
        `sum(g(Z) * exp(-eta * (r - rS)**2) * cutoff(r, rC), dim=1)`
    """
    return (g(Z) * torch.exp(-eta * (r - rS)**2) * cutoff(r, rC)).sum(dim=1, keepdim=True)
