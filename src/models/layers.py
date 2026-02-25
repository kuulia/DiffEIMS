import torch
import torch.nn as nn


class Xtoy(nn.Module):
    def __init__(self, dx, dy, device=None, dtype=None):
        """ Map node features to global features """
        kw = {'device': device, 'dtype': dtype}
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy, **kw)
        self.dtype = dtype

    def forward(self, X, x_mask):
        """ X: bs, n, dx. """
        X = X.to(self.dtype)
        x_mask = x_mask.to(self.dtype)
        x_mask = x_mask.expand(-1, -1, X.shape[-1])
        float_imask = 1 - x_mask
        m = X.sum(dim=1) / torch.sum(x_mask, dim=1)
        mi = (X + 1e5 * float_imask).min(dim=1)[0]
        ma = (X - 1e5 * float_imask).max(dim=1)[0]
        std = torch.sum(((X - m[:, None, :]) ** 2) * x_mask, dim=1) / torch.sum(x_mask, dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy, device=None, dtype=None):
        """ Map edge features to global features. """
        kw = {'device': device, 'dtype': dtype}
        super().__init__()
        self.lin = nn.Linear(4 * d, dy, **kw)
        self.dtype = dtype

    def forward(self, E, e_mask1, e_mask2):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        E = E.to(self.dtype)
        e_mask1 = e_mask1.to(self.dtype)
        e_mask2 = e_mask2.to(self.dtype)
        mask = (e_mask1 * e_mask2).expand(-1, -1, -1, E.shape[-1])
        float_imask = 1 - mask.to(self.dtype)
        divide = torch.sum(mask, dim=(1, 2))
        m = E.sum(dim=(1, 2)) / divide
        mi = (E + 1e5 * float_imask).min(dim=2)[0].min(dim=1)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0].max(dim=1)[0]
        std = torch.sum(((E - m[:, None, None, :]) ** 2) * mask, dim=(1, 2)) / divide
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


def masked_softmax(x, mask, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)