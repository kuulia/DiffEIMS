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
        X = X.to(self.dtype)

        x_mask = x_mask.bool()
        mask_float = x_mask.to(self.dtype)

        denom = mask_float.sum(dim=1) + 1e-6
        m = (X * mask_float).sum(dim=1) / denom

        float_imask = (~x_mask).to(self.dtype)

        mi = (X + 1e5 * float_imask).min(dim=1)[0]
        ma = (X - 1e5 * float_imask).max(dim=1)[0]

        std = (((X - m[:, None, :]) ** 2) * mask_float).sum(dim=1) / denom

        z = torch.cat((m, mi, ma, std), dim=-1)
        return self.lin(z)


class Etoy(nn.Module):
    def __init__(self, d, dy, device=None, dtype=None):
        """ Map edge features to global features. """
        kw = {'device': device, 'dtype': dtype}
        super().__init__()
        self.lin = nn.Linear(4 * d, dy, **kw)
        self.dtype = dtype

    def forward(self, E, e_mask1, e_mask2):
        """
        E: bs, n, n, de
        """

        # Ensure feature dtype
        E = E.to(self.dtype)

        # Build boolean mask
        mask = (e_mask1 * e_mask2).expand(-1, -1, -1, E.shape[-1]).bool()

        # Float version for arithmetic
        mask_float = mask.to(self.dtype)

        # Safe denominator (avoid divide-by-zero)
        denom = mask_float.sum(dim=(1, 2)) + 1e-6

        # Masked mean (IMPORTANT: mask before summing)
        m = (E * mask_float).sum(dim=(1, 2)) / denom

        # Inverted mask for min/max
        float_imask = (~mask).to(self.dtype)

        mi = (E + 1e5 * float_imask).min(dim=2)[0].min(dim=1)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0].max(dim=1)[0]

        # Masked variance
        std = (((E - m[:, None, None, :]) ** 2) * mask_float).sum(dim=(1, 2)) / denom

        z = torch.cat((m, mi, ma, std), dim=-1)

        return self.lin(z)


def masked_softmax(x, mask, **kwargs):
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    x = x.masked_fill(mask == 0, float("-inf"))
    return torch.softmax(x, **kwargs)