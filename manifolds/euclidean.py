"""Euclidean manifold."""

from manifolds.base import Manifold
import torch


class Euclidean(Manifold):
    """
    Euclidean Manifold class.
    """

    def __init__(self):
        super().__init__()
        self.name = 'Euclidean'

    def normalize(self, p):
        dim = p.size(-1)
        # p.view(-1, dim).renorm_(2, 0, 1.)
        return p.view(-1, dim).renorm(2, 0, 1.)  # Removed in-place operation

    def sqdist(self, p1, p2, c):
        return (p1 - p2).pow(2).sum(dim=-1)

    def egrad2rgrad(self, p, dp, c):
        return dp

    def proj(self, p, c):
        return p

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        return p + u

    def logmap(self, p1, p2, c):
        return p2 - p1

    def expmap0(self, u, c):
        return u

    def logmap0(self, p, c):
        return p

    def mobius_add(self, x, y, c, dim=-1):
        return x + y

    def mobius_matvec(self, m, x, c):
        mx = x @ m.transpose(-1, -2)
        return mx

    def init_weights(self, w, c, irange=1e-5):
        # w.data.uniform_(-irange, irange)
        distribution = torch.distributions.Uniform(-irange, irange)
        w = distribution.sample(w.shape)
        return w

    def inner(self, p, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, v, c):
        return v

    def ptransp0(self, x, v, c):
        return x + v
