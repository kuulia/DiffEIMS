from collections import Counter
from typing import List

import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from torchmetrics import Metric

from src.utils import is_valid, canonical_mol_from_inchi

# All metrics below subclass torchmetrics.Metric so their state is registered via
# add_state(..., dist_reduce_fx="sum"). Under DDP that makes compute() all-reduce
# the counters across ranks, giving a true global value rather than a per-rank one.
# The collections subclass nn.Module and hold their children in an nn.ModuleDict so
# the metric states are registered as submodules and follow the LightningModule onto
# the GPU.


# k values the top-k metrics are reported at. Reporting every k in 1..N is
# expensive twice over: each k is a separate Metric object, so under DDP each one
# adds its own all-reduce to compute(), and K_TanimotoSimilarity/K_CosineSimilarity
# rescore generated_mols[:k] from scratch on every update, making the fingerprint
# work sum to O(N^2) per molecule. It also buys nothing — top-63 and top-64 are the
# same number to 3 decimals. Dense at the low end, where adding one candidate still
# moves the ranking; sparse above it.
_TOP_K_REPORTED = tuple(range(1, 11)) + (20, 30, 40, 50, 100)


def top_k_list(num_samples: int) -> List[int]:
    """k values to report top-k metrics at, given `num_samples` generated candidates.

    k > num_samples is dropped: those metrics would be identical to top-num_samples.
    """
    return [k for k in _TOP_K_REPORTED if k <= num_samples]


class K_ACC(Metric):
    """
    Top-K Accuracy metric for molecule generation tasks.

    This metric measures how often the correct (true) InChI is among the
    top-k generated InChIs. It is commonly used in generative models to evaluate
    retrieval or ranking performance.

    Attributes:
        k (int): The number of top predictions to consider.
        correct (Tensor): Number of correct predictions so far (DDP-reduced).
        total (Tensor): Total number of evaluated examples (DDP-reduced).
    """

    full_state_update = False

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, generated_inchis: list[str], true_inchi: str):
        if true_inchi in generated_inchis[: self.k]:
            self.correct += 1
        self.total += 1

    def compute(self):
        if self.total == 0:
            return torch.zeros_like(self.correct)
        return self.correct / self.total


class K_ACC_Collection(nn.Module):
    def __init__(self, k_list: List[int]):
        super().__init__()
        self.k_list = k_list
        self.metrics = nn.ModuleDict({f"acc_at_{k}": K_ACC(k) for k in self.k_list})

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def update(self, generated_mols: list[str], true_mol: str):
        # filter mols and select unique
        inchis = [Chem.MolToInchi(mol) for mol in generated_mols if is_valid(mol)]

        inchi_counter = Counter(inchis)
        inchis = [item for item, count in inchi_counter.most_common()]

        true_inchi = Chem.MolToInchi(true_mol)
        for metric in self.metrics.values():
            metric.update(inchis, true_inchi)

    def compute(self):
        res = {}
        for k, metric in self.metrics.items():
            res[k] = metric.compute()
        return res


class K_TanimotoSimilarity(Metric):
    full_state_update = False

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.add_state("similarity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, generated_mols, true_mol):
        true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2, nBits=2048)

        max_sim = 0
        for mol in generated_mols[: self.k]:
            try:
                gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                max_sim = max(max_sim, DataStructs.TanimotoSimilarity(gen_fp, true_fp))
            except Exception as e:
                pass

        self.similarity += max_sim
        self.total += 1

    def compute(self):
        if self.total == 0:
            return torch.zeros_like(self.similarity)
        return self.similarity / self.total


class K_CosineSimilarity(Metric):
    full_state_update = False

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.add_state("similarity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, generated_mols, true_mol):
        true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2, nBits=2048)

        max_sim = 0
        for mol in generated_mols[: self.k]:
            try:
                gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                max_sim = max(max_sim, DataStructs.CosineSimilarity(gen_fp, true_fp))
            except Exception as e:
                pass

        self.similarity += max_sim
        self.total += 1

    def compute(self):
        if self.total == 0:
            return torch.zeros_like(self.similarity)
        return self.similarity / self.total


class K_SimilarityCollection(nn.Module):
    def __init__(self, k_list: List[int]):
        super().__init__()
        self.k_list = k_list
        metrics = {}
        for k in self.k_list:
            metrics[f"tanimoto_at_{k}"] = K_TanimotoSimilarity(k)
            metrics[f"cosine_at_{k}"] = K_CosineSimilarity(k)
        self.metrics = nn.ModuleDict(metrics)

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def update(self, generated_mols, true_mol):
        # filter mols and select unique
        inchis = [Chem.MolToInchi(mol) for mol in generated_mols if is_valid(mol)]

        inchi_counter = Counter(inchis)
        inchis = [item for item, count in inchi_counter.most_common()]

        processed_mols = [canonical_mol_from_inchi(inchi) for inchi in inchis]

        for metric in self.metrics.values():
            metric.update(processed_mols, true_mol)

    def compute(self):
        res = {}
        for k, metric in self.metrics.items():
            res[k] = metric.compute()
        return res


class Validity(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("valid", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, generated_mols):
        for mol in generated_mols:
            if is_valid(mol):
                self.valid += 1
            self.total += 1

    def compute(self):
        if self.total == 0:
            return torch.zeros_like(self.valid)
        return self.valid / self.total


class MeanTanimotoSimilarity(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        # Running sum rather than a Python list: a list cannot be DDP-reduced,
        # and sum/count is equivalent for a mean.
        self.add_state("similarity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred_fp, true_fp):
        """
        Add a Tanimoto similarity score between two fingerprints.
        Args:
            pred_fp: RDKit ExplicitBitVect or compatible
            true_fp: RDKit ExplicitBitVect
        """
        try:
            sim = DataStructs.TanimotoSimilarity(pred_fp, true_fp)
            self.similarity += sim
            self.total += 1
        except Exception as e:
            # Could log the error if needed
            pass

    def compute(self):
        if self.total == 0:
            return torch.zeros_like(self.similarity)
        return self.similarity / self.total
