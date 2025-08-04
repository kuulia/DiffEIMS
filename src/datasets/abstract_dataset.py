"""
This module defines dataset and datamodule classes used for training and evaluating
the DiffMS model using PyTorch Lightning and PyTorch Geometric.

Key Components:
- `CustomLightningDataset`: Wraps training/validation/test datasets with custom 
   dataloader behavior, compatible with PyTorch Lightning.
- `AbstractDataModule`: Adds statistical analysis tools for dataset inspection.
- `MolecularDataModule`: Adds chemical valency distribution computation.
- `AbstractDatasetInfos`: Extracts and stores input/output dimensionalities from a 
   datamodule, including augmented features.
- GLOBALS: `ATOM_TO_VALENCY`, `ATOM_TO_WEIGHT`, `ATOM_DECODER`: Lookup tables for atoms.
"""
import copy

import torch
import pytorch_lightning as pl
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data.lightning import LightningDataset
from torch.utils.data import IterableDataset

import src.utils as utils
from src.diffusion.distributions import DistributionNodes

def kwargs_repr(**kwargs) -> str:
    """util function to format keyword arguments as a string."""
    return ', '.join([f'{k}={v}' for k, v in kwargs.items() if v is not None])

class CustomLightningDataset(LightningDataset):
    """
    A wrapper over PyTorch Geometric's LightningDataset that allows for flexible
    dataloader configurations, including different batch sizes for train and eval.

    Attributes:
        batch_size (int): Training batch size.
        eval_batch_size (int): Evaluation batch size.
        kwargs (dict): Dataloader arguments (e.g., num_workers, pin_memory).
    """

    def __init__(self, cfg, datasets: dict, **kwargs):
        """
        Args:
            cfg: Hydra configuration.
            datasets (dict): Dictionary containing 'train', 'val', and 'test' datasets.
            **kwargs: Additional DataLoader arguments.
        """
        kwargs.pop('batch_size', None)
        self.kwargs = kwargs

        self.batch_size = cfg.train.batch_size if 'debug' not in cfg.general.name else 2
        self.eval_batch_size = cfg.train.eval_batch_size if 'debug' not in cfg.general.name else 1

        super().__init__(train_dataset=datasets['train'], val_dataset=datasets['val'], test_dataset=datasets['test'],)
        for k, v in kwargs.items(): # overwrite default kwargs from LightningDataset
            self.kwargs[k] = v
        self.kwargs.pop('batch_size', None)

    def dataloader(self, dataset: Dataset, **kwargs) -> DataLoader:
        '''return DataLoader(dataset, **kwargs)'''
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        '''
        Constructs the training dataloader with proper shuffling behavior.

        Shuffling is enabled by default, except when:
        - The dataset is an IterableDataset (e.g., for streaming or lazy loading),
        - A custom sampler is provided (which controls data ordering),
        - A custom batch_sampler is used (which defines batching behavior).
            return self.dataloader(self.train_dataset, shuffle=shuffle, batch_size=self.batch_size, **self.kwargs)
            '''
        # don't shuffle if...
        shuffle = not isinstance(self.train_dataset, IterableDataset) # is instance of IterableDataset: e.g. to avoid streaming/lazy loading incompatibility
        shuffle &= self.kwargs.get('sampler', None) is None # a custom sampler is provided
        shuffle &= self.kwargs.get('batch_sampler', None) is None # a custom batch sampler is provided
        return self.dataloader(self.train_dataset, shuffle=shuffle, batch_size=self.batch_size, **self.kwargs)

    def val_dataloader(self) -> DataLoader:
        """
        Constructs the validation dataloader.

        Always uses `shuffle=True` to improve evaluation diversity when applicable.
        Removes 'sampler' and 'batch_sampler' from kwargs to avoid conflicts.

        Returns:
            DataLoader: Configured PyTorch Geometric dataloader for validation.

        return self.dataloader(self.val_dataset, shuffle=True, batch_size=self.eval_batch_size, **kwargs)
        """
        kwargs = copy.copy(self.kwargs)
        kwargs.pop('sampler', None)
        kwargs.pop('batch_sampler', None)

        return self.dataloader(self.val_dataset, shuffle=True, batch_size=self.eval_batch_size, **kwargs)

    def test_dataloader(self) -> DataLoader:
        """
        Constructs the test dataloader.

        Always uses `shuffle=False` to ensure deterministic evaluation.
        Removes 'sampler' and 'batch_sampler' from kwargs to avoid conflicts.

        Returns:
            DataLoader: Configured PyTorch Geometric dataloader for testing.
        """
        kwargs = copy.copy(self.kwargs)
        kwargs.pop('sampler', None)
        kwargs.pop('batch_sampler', None)

        return self.dataloader(self.test_dataset, shuffle=False,  batch_size=self.eval_batch_size, **kwargs)

    def predict_dataloader(self) -> DataLoader:
        """
        Constructs the prediction dataloader.

        Always uses `shuffle=False` to preserve input order during inference.
        Removes 'sampler' and 'batch_sampler' from kwargs to avoid conflicts.

        Returns:
            DataLoader: Configured PyTorch Geometric dataloader for prediction.
        """
        kwargs = copy.copy(self.kwargs)
        kwargs.pop('sampler', None)
        kwargs.pop('batch_sampler', None)

        return self.dataloader(self.pred_dataset, shuffle=False,  batch_size=self.eval_batch_size, **kwargs)

    def __repr__(self) -> str:
        '''
        developer printing kwargs with kwargs_repr()
        '''
        kwargs = kwargs_repr(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            test_dataset=self.test_dataset,
            pred_dataset=self.pred_dataset, 
            batch_size=self.batch_size,
            eval_batch_size=self.eval_batch_size,
            **self.kwargs
        )
        
        return f'{self.__class__.__name__}({kwargs})'

class AbstractDataModule(CustomLightningDataset):
    """
    Extension of `CustomLightningDataset` that provides utility methods for analyzing 
    node and edge distributions in graph datasets.

    Attributes:
        cfg: Hydra config object.
        input_dims (Optional[int]): Dimensionality of node features (can be inferred).
        output_dims (Optional[int]): Dimensionality of output/labels (can be inferred).
    """
    def __init__(self, cfg, datasets: dict):
        """
        Args:
            cfg: Hydra configuration containing training parameters.
            datasets (dict): Dictionary with 'train', 'val', and 'test' datasets.
        """
        super().__init__(cfg, datasets, num_workers=cfg.train.num_workers, pin_memory=getattr(cfg.train, "pin_memory", True))
        self.cfg = cfg
        self.input_dims = None
        self.output_dims = None

    def __getitem__(self, idx):
        """Allows direct indexing into the training dataset."""
        return self.train_dataset[idx]

    def node_counts(self, max_nodes_possible=150):
        """
        Computes a normalized histogram of graph sizes (number of nodes per graph)
        from the training and validation datasets.

        This method iterates through the training and validation dataloaders, counts the
        number of nodes in each individual graph within the batches, accumulates these counts
        into a histogram, and normalizes the histogram to form a probability distribution.

        Args:
            max_nodes_possible (int): The maximum number of nodes considered in the
            histogram. Graphs with node counts equal to or exceeding this value are
            ignored to limit the histogram size. Default is 150.

        Returns:
            torch.Tensor: A 1D tensor representing the normalized histogram of node counts.
                Each index `i` corresponds to the relative frequency of graphs with `i` nodes.
                The tensor length corresponds to the maximum observed graph size (up to
                `max_nodes_possible`).
        """
        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for data in loader:
                _, counts = torch.unique(data.batch, return_counts=True) # Get number of nodes per graph in the batch
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum() # normalize
        return all_counts

    def node_types(self):
        """
        Computes the normalized distribution of node types (features) across the training dataset.

        This method assumes that node features (`data.x`) are one-hot encoded or
        multi-hot encoded vectors, where each dimension corresponds to a specific
        node type or feature class. It sums the occurrences of each node type
        across all graphs in the training set and normalizes the result to obtain
        relative frequencies.

        Returns:
            torch.Tensor: A 1D tensor of length equal to the number of node feature
                dimensions (`num_classes`), where each value represents the relative
                frequency of that node type in the training data. The tensor sums to 1.

        Notes:
            - Assumes `data.x` has shape `(num_nodes, num_node_features)` where
            each row is a feature vector for a node.
            - The normalization divides by the total count of all node features summed,
            so the output reflects proportions.
            - Only the training dataset is used for this calculation.
         """
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.x.shape[1]
            break

        counts = torch.zeros(num_classes)

        for i, data in enumerate(self.train_dataloader()):
            counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        """
        Computes the normalized distribution of edge types, including "non-edges",
        across the training dataset.

        This method assumes:
        - `data.edge_attr` is a one-hot or multi-hot encoded tensor representing
            edge types, where dimension 0 corresponds to "non-edge" (no bond).
        - The batch contains multiple graphs, with node-to-graph assignments
            specified by `data.batch`.
        - The complete graph for each subgraph contains all possible node pairs.

        The method calculates:
        1. The total possible pairs of nodes (all pairs within each graph).
        2. The actual number of edges present.
        3. The number of "non-edges" (pairs without an edge) as difference.
        4. The sum of each edge type over the batch.
        5. Aggregates and normalizes the counts into a probability distribution.

        Returns:
            torch.Tensor: A 1D tensor where each index corresponds to the relative frequency
                of an edge type. Index 0 represents the "non-edge" class, while indices
                1 and onwards represent actual edge types. The tensor sums to 1.

        Notes:
            - This method uses the training dataset only.
            - The method assumes undirected graphs without self-loops; if not,
            interpretation of counts may differ.
            - The assertion `num_non_edges >= 0` ensures the integrity of counts.
        """
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.zeros(num_classes, dtype=torch.float)

        for _, data in enumerate(self.train_dataloader()):
            _, counts = torch.unique(data.batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

        d = d / d.sum()
        return d


class MolecularDataModule(AbstractDataModule):
    """
    Data module specialized for molecular graph datasets, extending `AbstractDataModule`
    (which itself extends `CustomLightningDataset`).

    Provides molecular-specific data statistics and utilities, such as valency
    distribution calculation, tailored for chemical graphs.

    Inherits:
        - AbstractDataModule: Provides base functionality for node, edge statistics
          and dataset loading.
        - CustomLightningDataset: Handles PyTorch Lightning dataloaders with
          flexible batch and shuffle configurations.
    """

    def valency_count(self, max_n_nodes):
        """
        Computes a normalized histogram of atomic valencies across the training dataset.

        The valency of each atom is computed by summing the weighted counts of its bonds.
        Bond types considered include: no bond, single, double, triple, and aromatic bonds,
        each assigned a multiplier corresponding to their bond order.

        Args:
            max_n_nodes (int): Maximum number of nodes (atoms) in a graph to estimate
                the maximum possible valency histogram size. The size of the output
                tensor is `3 * max_n_nodes - 2`, which is an upper bound on possible valencies.

        Returns:
            torch.Tensor: A normalized 1D tensor representing the distribution of atomic
                valencies observed in the training dataset. Each index corresponds to a
                valency value, and the sum over all indices is 1.

        Details:
            - Assumes `data.x.shape[0]` is the number of atoms (nodes) in the graph.
            - `data.edge_index[0] == atom` filters all edges originating from a specific atom.
            - `data.edge_attr` encodes bond types in a one-hot or multi-hot format per edge.
            - Multipliers correspond to the effective bond order to calculate valency:
              0: no bond, 1: single bond, 2: double bond, 3: triple bond, 1.5: aromatic bond.
            - Valencies are accumulated across all atoms and normalized to form a probability
              distribution.
        """
        valencies = torch.zeros(3 * max_n_nodes - 2)   # Max valency possible if everything is connected

        # No bond, single bond, double bond, triple bond, aromatic bond
        multiplier = torch.tensor([0, 1, 2, 3, 1.5])

        for data in self.train_dataloader():
            n = data.x.shape[0]

            for atom in range(n):
                edges = data.edge_attr[data.edge_index[0] == atom]
                edges_total = edges.sum(dim=0)
                valency = (edges_total * multiplier).sum()
                valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies


class AbstractDatasetInfos:
    """
    Utility class for computing and storing dataset-related metadata such as
    input/output dimensionalities, number of classes, and node distribution.
    This is to help ensure the model architechture is properly configured and
    keeps consistent dimensional info centralized

    Attributes:
        input_dims (dict or None): Dictionary with keys 'X', 'E', 'y' representing
            input feature dimensions for nodes (X), edges (E), and targets (y).
        output_dims (dict or None): Dictionary with keys 'X', 'E', 'y' representing
            output feature dimensions for nodes, edges, and targets.
        num_classes (int): Number of node classes/types in the dataset.
        max_n_nodes (int): Maximum number of nodes in any graph, inferred from node counts.
        nodes_dist (DistributionNodes): Distribution object representing node count statistics.
    """
    
    def complete_infos(self, n_nodes, node_types):
        """
        Finalizes dataset metadata by setting:
          - number of node classes,
          - maximum node count,
          - node distribution.

        Args:
            n_nodes (torch.Tensor or list): Histogram or counts of node numbers per graph.
            node_types (torch.Tensor or list): Distribution or counts of node types/classes.
        """
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        """
        Computes input and output feature dimensions for the dataset by analyzing
        a sample batch from the provided data module and applying extra feature
        extraction functions.

        Args:
            datamodule (AbstractDataModule): Data module providing dataloaders.
            extra_features (callable): Function that extracts extra features from
                a batch dictionary, returning an object with attributes X, E, y.
            domain_features (callable): Function that extracts domain-specific features
                similarly returning an object with attributes X, E, y.

        Sets:
            self.input_dims (dict): Sum of base feature dimensions and extra features for
                nodes ('X'), edges ('E'), and targets ('y'). The target dimension
                includes a +1 offset due to time conditioning.
            self.output_dims (dict): Base feature dimensions of nodes, edges, and targets
                (without extra features).

        Notes:
            - Uses `utils.to_dense` to convert sparse batch data into dense representations.
            - The `example_data` dictionary packages dense node and edge features,
              target values, and node masks for feature extraction functions.
        """
        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = utils.to_dense(example_batch.x, example_batch.edge_index, example_batch.edge_attr,
                                             example_batch.batch)
        example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': example_batch['y'], 'node_mask': node_mask}

        self.input_dims = {'X': example_batch['x'].size(1),
                           'E': example_batch['edge_attr'].size(1),
                           'y': example_batch['y'].size(1) + 1}      # + 1 due to time conditioning

        ex_extra_feat = extra_features(example_data)
        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {'X': example_batch['x'].size(1),
                            'E': example_batch['edge_attr'].size(1),
                            'y': example_batch['y'].size(1)}


ATOM_TO_VALENCY = {
    'H': 1,
    'He': 0,
    'Li': 1,
    'Be': 2,
    'B': 3,
    'C': 4,
    'N': 3,
    'O': 2,
    'F': 1,
    'Ne': 0,
    'Na': 1,
    'Mg': 2,
    'Al': 3,
    'Si': 4,
    'P': 3,
    'S': 2,
    'Cl': 1,
    'Ar': 0,
    'K': 1,
    'Ca': 2,
    'Sc': 3,
    'Ti': 4,
    'V': 5,
    'Cr': 2,
    'Mn': 7,
    'Fe': 2,
    'Co': 3,
    'Ni': 2,
    'Cu': 2,
    'Zn': 2,
    'Ga': 3,
    'Ge': 4,
    'As': 3,
    'Se': 2,
    'Br': 1,
    'Kr': 0,
    'Rb': 1,
    'Sr': 2,
    'Y': 3,
    'Zr': 2,
    'Nb': 2,
    'Mo': 2,
    'Tc': 6,
    'Ru': 2,
    'Rh': 3,
    'Pd': 2,
    'Ag': 1,
    'Cd': 1,
    'In': 1,
    'Sn': 2,
    'Sb': 3,
    'Te': 2,
    'I': 1,
    'Xe': 0,
    'Cs': 1,
    'Ba': 2,
    'La': 3,
    'Ce': 3,
    'Pr': 3,
    'Nd': 3,
    'Pm': 3,
    'Sm': 2,
    'Eu': 2,
    'Gd': 3,
    'Tb': 3,
    'Dy': 3,
    'Ho': 3,
    'Er': 3,
    'Tm': 2,
    'Yb': 2,
    'Lu': 3,
    'Hf': 4,
    'Ta': 3,
    'W': 2,
    'Re': 1,
    'Os': 2,
    'Ir': 1,
    'Pt': 1,
    'Au': 1,
    'Hg': 1,
    'Tl': 1,
    'Pb': 2,
    'Bi': 3,
    'Po': 2,
    'At': 1,
    'Rn': 0,
    'Fr': 1,
    'Ra': 2,
    'Ac': 3,
    'Th': 4,
    'Pa': 5,
    'U': 2,
}

ATOM_TO_WEIGHT = {
    'H': 1,
    'He': 4,
    'Li': 7,
    'Be': 9,
    'B': 11,
    'C': 12,
    'N': 14,
    'O': 16,
    'F': 19,
    'Ne': 20,
    'Na': 23,
    'Mg': 24,
    'Al': 27,
    'Si': 28,
    'P': 31,
    'S': 32,
    'Cl': 35,
    'Ar': 40,
    'K': 39,
    'Ca': 40,
    'Sc': 45,
    'Ti': 48,
    'V': 51,
    'Cr': 52,
    'Mn': 55,
    'Fe': 56,
    'Co': 59,
    'Ni': 59,
    'Cu': 64,
    'Zn': 65,
    'Ga': 70,
    'Ge': 73,
    'As': 75,
    'Se': 79,
    'Br': 80,
    'Kr': 84,
    'Rb': 85,
    'Sr': 88,
    'Y': 89,
    'Zr': 91,
    'Nb': 93,
    'Mo': 96,
    'Tc': 98,
    'Ru': 101,
    'Rh': 103,
    'Pd': 106,
    'Ag': 108,
    'Cd': 112,
    'In': 115,
    'Sn': 119,
    'Sb': 122,
    'Te': 128,
    'I': 127,
    'Xe': 131,
    'Cs': 133,
    'Ba': 137,
    'La': 139,
    'Ce': 140,
    'Pr': 141,
    'Nd': 144,
    'Pm': 145,
    'Sm': 150,
    'Eu': 152,
    'Gd': 157,
    'Tb': 159,
    'Dy': 163,
    'Ho': 165,
    'Er': 167,
    'Tm': 169,
    'Yb': 173,
    'Lu': 175,
    'Hf': 178,
    'Ta': 181,
    'W': 184,
    'Re': 186,
    'Os': 190,
    'Ir': 192,
    'Pt': 195,
    'Au': 197,
    'Hg': 201,
    'Tl': 204,
    'Pb': 207,
    'Bi': 209,
    'Po': 209,
    'At': 210,
    'Rn': 222,
    'Fr': 223,
    'Ra': 226,
    'Ac': 227,
    'Th': 232,
    'Pa': 231,
    'U': 238,
    'Np': 237,
    'Pu': 244,
    'Am': 243,
    'Cm': 247,
    'Bk': 247,
    'Cf': 251,
    'Es': 252,
    'Fm': 257,
    'Md': 258,
    'No': 259,
    'Lr': 262,
    'Rf': 267,
    'Db': 270,
    'Sg': 269,
    'Bh': 264,
    'Hs': 269,
    'Mt': 278,
    'Ds': 281,
    'Rg': 282,
    'Cn': 285,
    'Nh': 286,
    'Fl': 289,
    'Mc': 290,
    'Lv': 293,
    'Ts': 294,
    'Og': 294,
}

ATOM_DECODER = ('C', 'O', 'P', 'N', 'S', 'Cl', 'F', 'H')