import math
import numpy as np
import torch
import torch.nn.functional as F


def get_posenc(edge_index, num_posenc=16):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    num_posenc = num_posenc
    d = edge_index[0] - edge_index[1]
    
    frequency = torch.exp(
        torch.arange(0, num_posenc, 2, dtype=torch.float32)
        * -(np.log(10000.0) / num_posenc)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def get_orientations(X):
    # X : num_conf x num_res x 3
    forward = normalize(X[:, 1:] - X[:, :-1])
    backward = normalize(X[:, :-1] - X[:, 1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def get_sidechains(X):
    # X : num_conf x num_res x 3 x 3
    p, origin, n = X[:, :, 0], X[:, :, 1], X[:, :, 2]
    n, p = normalize(n - origin), normalize(p - origin)
    return torch.cat([n.unsqueeze_(-2), p.unsqueeze_(-2)], -2)


def normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.linalg.norm(tensor, dim=dim, keepdim=True)))


def rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    
    TODO switch to DimeNet RBFs
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF
