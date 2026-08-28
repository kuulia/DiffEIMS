import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric, MeanSquaredError


class TrainAbstractMetricsDiscrete(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, log: bool):
        pass

    def reset(self):
        pass

    def log_epoch_metrics(self):
        return None, None


class TrainAbstractMetrics(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        masked_pred_epsX,
        masked_pred_epsE,
        pred_y,
        true_epsX,
        true_epsE,
        true_y,
        log,
    ):
        pass

    def reset(self):
        pass

    def log_epoch_metrics(self):
        return None, None


class SumExceptBatchMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, values) -> None:
        self.total_value += torch.sum(values)
        self.total_samples += values.shape[0]

    def compute(self):
        return self.total_value / self.total_samples


class SumExceptBatchMSE(MeanSquaredError):
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        assert preds.shape == target.shape
        sum_squared_error, n_obs = self._mean_squared_error_update(preds, target)

        self.sum_squared_error += sum_squared_error
        self.total += n_obs

    def _mean_squared_error_update(self, preds: Tensor, target: Tensor):
        """Updates and returns variables required to compute Mean Squared Error. Checks for same shape of input
        tensors.
            preds: Predicted tensor
            target: Ground truth tensor
        """
        diff = preds - target
        sum_squared_error = torch.sum(diff * diff)
        n_obs = preds.shape[0]
        return sum_squared_error, n_obs


class SumExceptBatchKL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, p, q) -> None:
        self.total_value += F.kl_div(q, p, reduction="sum")
        self.total_samples += p.size(0)

    def compute(self):
        return self.total_value / self.total_samples


class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_ce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.
        preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
        target: Ground truth values     (bs * n, d) or (bs * n * n, d)."""
        target = torch.argmax(target, dim=-1)
        output = F.cross_entropy(preds, target, reduction="sum")
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples


class FingerprintBCEMetric(Metric):
    """Per-bit BCE between predicted and true molecular fingerprints.

    This is MIST's fingerprint objective (mist/models/mist_model.py: loss_fn
    defaults to "bce", nn.BCELoss(reduction="none") then .mean(-1)), adapted for
    one difference in this repo: MIST's FPGrowingModule ends in nn.Sigmoid, so its
    predictions are probabilities and plain BCELoss is correct. Here the head that
    produces a 2048-bit prediction is merge_function = nn.Linear(4096, morgan_nbits)
    with no activation, so the input is logits and the numerically stable
    with_logits form is the faithful equivalent.

    Contrast CrossEntropyMetric, which does target.argmax(-1) and so collapses a
    multi-hot fingerprint to a single class index -- the first set bit. That is why
    val/y_CE sat at ~7.4 against ln(2048) = 7.62, i.e. chance.
    """

    def __init__(self):
        super().__init__()
        self.add_state("total_bce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """preds: fingerprint logits (bs, n_bits); target: 0/1 bits (bs, n_bits)."""
        # mean over bits, sum over batch -- so the reported value is per molecule
        # and independent of n_bits, matching MIST's fp_loss_full.mean(-1).
        per_mol = F.binary_cross_entropy_with_logits(
            preds, target.float(), reduction="none"
        ).mean(dim=-1)
        self.total_bce += per_mol.sum()
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_bce / self.total_samples


class ProbabilityMetric(Metric):
    def __init__(self):
        """This metric is used to track the marginal predicted probability of a class during training."""
        super().__init__()
        self.add_state("prob", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor) -> None:
        self.prob += preds.sum()
        self.total += preds.numel()

    def compute(self):
        return self.prob / self.total


class NLL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_nll", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch_nll) -> None:
        self.total_nll += torch.sum(batch_nll)
        self.total_samples += batch_nll.numel()

    def compute(self):
        return self.total_nll / self.total_samples
