import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanSquaredError
import wandb
from src.metrics.abstract_metrics import CrossEntropyMetric, FingerprintBCEMetric


class TrainLossDiscrete(nn.Module):
    """Train with Cross entropy"""

    def __init__(self, lambda_train, iterative_loss_weight: float = 0.0):
        super().__init__()
        # MIST's "growing" auxiliary: supervise every FPGrowingModule intermediate
        # against the true fingerprint folded down to that intermediate's width.
        # Library default is 0.0, but MIST's shipped configs use 0.6 (csi_fp_mist)
        # and 0.4 (canopus_fp_mist), so 0 is a debug default, not a real one.
        self.iterative_loss_weight = iterative_loss_weight
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        # Fingerprint prediction is multi-label over morgan_nbits independent bits,
        # so it needs per-bit BCE (MIST's default loss_fn), not softmax CE over an
        # argmax'd target. Only active when lambda_train[2] != 0, i.e. encoder
        # training; e2e uses [1, 1, 0] and never reaches it.
        self.y_loss = FingerprintBCEMetric()
        self.lambda_train = lambda_train

    def forward(
        self,
        masked_pred_X,
        masked_pred_E,
        pred_y,
        true_X,
        true_E,
        true_y,
        log: bool,
        int_preds=None,
    ):
        """Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, n_bits) predicted fingerprint LOGITS
        true_y : tensor -- (bs, n_bits) true fingerprint bits (0/1)
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        log : boolean."""
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(
            masked_pred_X, (-1, masked_pred_X.size(-1))
        )  # (bs * n, dx)
        masked_pred_E = torch.reshape(
            masked_pred_E, (-1, masked_pred_E.size(-1))
        )  # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.0).any(dim=-1)
        mask_E = (true_E != 0.0).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        # Guarded on the weight, not just on emptiness: with merge='mist_fp',
        # pred_y is an FPGrowingModule intermediate whose width need not equal
        # morgan_nbits, so BCE would raise on shape. Every config that uses that
        # merge also sets lambda_train[2] = 0, so skipping is both correct and
        # avoids computing a term that is about to be multiplied by zero.
        compute_y = (
            self.lambda_train[2] != 0
            and true_y is not None
            and true_y.numel() > 0
            and pred_y.shape == true_y.shape
        )
        loss_y = self.y_loss(pred_y, true_y) if compute_y else 0.0

        # Iterative ("growing") auxiliary loss, ported from MIST's compute_loss.
        # int_preds are the FPGrowingModule intermediates, widths [2048, 1024, 512,
        # 256] once reversed for a 4096 head with refine_layers=4. Each is
        # supervised against the target folded to its width by taking the set bits
        # modulo that width, exactly as MIST does. Unlike the final head, these
        # come out of nn.Sigmoid bricks and are already probabilities, so they take
        # plain BCE rather than the with-logits form used for pred_y.
        iterative_loss = 0.0
        if compute_y and self.iterative_loss_weight and int_preds:
            cur_targ = true_y.float()
            aux_loss = None
            for int_pred in int_preds[::-1]:
                targ_shape = int_pred.shape[-1]
                batch_ind, bit_ind = torch.where(cur_targ)
                bit_ind = bit_ind % targ_shape
                new_targ = torch.zeros_like(int_pred.detach())
                new_targ[batch_ind, bit_ind] += 1
                new_targ = torch.clamp(new_targ, max=1)
                temp_loss = F.binary_cross_entropy(
                    int_pred, new_targ, reduction="none"
                ).mean(-1)
                aux_loss = temp_loss if aux_loss is None else aux_loss + temp_loss
                cur_targ = new_targ
            if aux_loss is not None:
                iterative_loss = self.iterative_loss_weight * aux_loss.mean()

        if log:
            to_log = {
                "train_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                "train_loss/X_CE": (
                    self.node_loss.compute() if true_X.numel() > 0 else -1
                ),
                "train_loss/E_CE": (
                    self.edge_loss.compute() if true_E.numel() > 0 else -1
                ),
                "train_loss/y_BCE": self.y_loss.compute() if compute_y else -1,
                "train_loss/iterative": (
                    iterative_loss.detach()
                    if torch.is_tensor(iterative_loss)
                    else iterative_loss
                ),
            }
            if wandb.run:
                wandb.log(to_log, commit=True)

            self.reset()

        return (
            self.lambda_train[0] * loss_X
            + self.lambda_train[1] * loss_E
            + self.lambda_train[2] * loss_y
            + iterative_loss
        )

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = (
            self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        )
        epoch_edge_loss = (
            self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        )
        epoch_y_loss = self.y_loss.compute() if self.y_loss.total_samples > 0 else -1

        to_log = {
            "train_epoch/x_CE": epoch_node_loss,
            "train_epoch/E_CE": epoch_edge_loss,
            "train_epoch/y_BCE": epoch_y_loss,
        }
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log
