import torch

class DistributionNodes:
    def __init__(self, histogram=None, N_max=100, lam=25.0):
        """
        Poisson-based node count prior.

        Args:
            histogram: optional (ignored for Poisson version, kept for compatibility)
            N_max: physical upper bound for nodes
            lam: Poisson rate parameter (if None, inferred or defaulted)
        """

        self.N_max = N_max

        self.lam = lam

        # --- build truncated Poisson over [0, N_max] ---
        k = torch.arange(0, N_max + 1, dtype=torch.float32)

        log_p = k * torch.log(torch.tensor(lam)) - lam - torch.lgamma(k + 1)

        p = torch.exp(log_p)

        # normalize over truncated support
        self.prob = p / p.sum()

    def sample_n(self, n_samples, device):
        dist = torch.distributions.Categorical(probs=self.prob.to(device))
        return dist.sample((n_samples,))

    def log_prob(self, batch_n_nodes):
        """
        Safe log-prob with physical bounding.
        """

        p = self.prob.to(batch_n_nodes.device)

        n = torch.clamp(batch_n_nodes, 0, self.N_max)

        return torch.log(p[n] + 1e-12)