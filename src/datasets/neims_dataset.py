from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
import numpy as np

class NeimsDataset(Dataset):
    def __init__(self, spectra_list, fingerprints, mol_graphs=None):
        self.spectra = spectra_list        # List of (m/z, intensity) arrays
        self.fingerprints = fingerprints   # List of fingerprint vectors
        self.mol_graphs = mol_graphs       # Optional

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectrum = self.spectra[idx]  # shape: (num_peaks, 2)
        fingerprint = self.fingerprints[idx]

        # Format for SpectraEncoderGrowing
        spectrum_tensor = torch.tensor(spectrum, dtype=torch.float)
        fingerprint_tensor = torch.tensor(fingerprint, dtype=torch.float)

        item = {
            "spectrum": spectrum_tensor,
            "y": fingerprint_tensor,
        }

        if self.mol_graphs:
            item["graph"] = self.mol_graphs[idx]

        return item