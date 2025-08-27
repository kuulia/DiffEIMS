import os
from typing import List, NoReturn, Tuple, Optional

import numpy as np
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch
import logging
import torch
import omegaconf
import wandb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect


def cfg_to_dict(cfg):
    return omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'graph_ddm_{cfg.dataset.name}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')

def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)

def is_valid(mol):
    smiles = mol2smiles(mol)
    if smiles is None:
        return False

    try:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except:
        return False
    if len(mol_frags) > 1:
        return False
    
    return True

def inchi_to_fingerprint(inchi: str, nbits: int = 2048, radius=3) -> np.ndarray:
    """get_morgan_fp."""

    mol = Chem.MolFromInchi(inchi)

    curr_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)

    fingerprint = np.zeros((0,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(curr_fp, fingerprint)
    return fingerprint

def tanimoto_sim(x: np.ndarray, y: np.ndarray) -> List[float]:
    # Calculate tanimoto distance with binary fingerprint
    intersect_mat = x & y
    union_mat = x | y

    intersection = intersect_mat.sum(-1)
    union = union_mat.sum(-1)

    ### I took the reciprocal here so instead of tanimoto sim, it became
    # distance. Could have just made negative but
    # sklearn doesn't accept negative distance matrices
    output = intersection / union
    return output

def cosine_sim(x: np.ndarray, y: np.ndarray) -> List[float]:
    # Calculate cosine similarity with binary fingerprint
    dot_product = np.dot(x, y)

    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    output = dot_product / (norm_x * norm_y)
    return output

try:
    from rdkit.Chem.MolStandardize.tautomer import TautomerCanonicalizer, TautomerTransform
    _RD_TAUTOMER_CANONICALIZER = 'v1'
    _TAUTOMER_TRANSFORMS = (
        TautomerTransform('1,3 heteroatom H shift',
                          '[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7,#16,#8,Se,Te]'),
        TautomerTransform('1,3 (thio)keto/enol r', '[O,S,Se,Te;X2!H0]-[C]=[C]'),
    )
except ModuleNotFoundError:
    from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator  # newer rdkit
    _RD_TAUTOMER_CANONICALIZER = 'v2'

def canonical_mol_from_inchi(inchi):
    """Canonicalize mol after Chem.MolFromInchi
    Note that this function may be 50 times slower than Chem.MolFromInchi"""
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return None
    if _RD_TAUTOMER_CANONICALIZER == 'v1':
        _molvs_t = TautomerCanonicalizer(transforms=_TAUTOMER_TRANSFORMS)
        mol = _molvs_t.canonicalize(mol)
    else:
        _te = TautomerEnumerator()
        mol = _te.Canonicalize(mol)
    return mol

def tensor_to_bitvect(t: torch.Tensor, threshold: float = 0.5) -> ExplicitBitVect:
    arr = (t >= threshold).int().cpu().numpy().tolist()
    bv = DataStructs.CreateFromBitString(''.join(str(b) for b in arr))
    return bv


def make_result_dirs(dirs_to_create: list[str]) -> NoReturn:

    for path in dirs_to_create:
        os.makedirs(path, exist_ok=True)

def get_nonstatic_cfg_params(cfg: omegaconf.DictConfig) -> Tuple[str, str, str, str]:
    """Get only important, non-static parts of the config."""
    
    if getattr(cfg.dataset, "augment_data", False):
        params = {
            "dataset": ["name", "remove_h", "augment_data", "remove_prob", "remove_weights",
                        "inten_prob", "inten_transform", "set_pooling", "collated_pkl_file",
                        "inference_only", "override_prev_dataset_cfg"],
            "general": ["decoder", "encoder", "resume", "test_only", "load_weights",
                        "encoder_finetune_strategy", "decoder_finetune_strategy",
                        "val_samples_to_generate", "test_samples_to_generate",
                        "num_test_samples"],
            "train": ["n_epochs", "batch_size", "eval_batch_size", "lr", 
                      "optimizer", "scheduler", "limit_val_batches"],
            "model": ["lambda_train"]
        }
    else:
        params = {
            "dataset": ["name", "remove_h", "augment_data", "inten_transform", 
                        "set_pooling", "collated_pkl_file", "inference_only", 
                        "override_prev_dataset_cfg"],
            "general": ["decoder", "encoder", "resume", "test_only", "load_weights",
                        "encoder_finetune_strategy", "decoder_finetune_strategy",
                        "val_samples_to_generate", "test_samples_to_generate",
                        "num_test_samples"],
            "train": ["n_epochs", "batch_size", "eval_batch_size", "lr", 
                      "optimizer", "scheduler", "limit_val_batches"],
            "model": ["lambda_train"]
        }

    def extract_section(section: str):
        return {
            k: getattr(cfg[section], k) for k in params[section] if k in cfg[section]
        }

    dataset_cfg = OmegaConf.to_yaml(extract_section("dataset"), resolve=True)
    general_cfg = OmegaConf.to_yaml(extract_section("general"), resolve=True)
    train_cfg   = OmegaConf.to_yaml(extract_section("train"), resolve=True)
    model_cfg   = OmegaConf.to_yaml(extract_section("model"), resolve=True)

    return dataset_cfg, general_cfg, train_cfg, model_cfg


def log_nonstatic_cfg(cfg, *, 
                      logger: Optional[logging.Logger] = None) -> NoReturn:
    """
    Logs the important, non-static parts of a configuration object in a structured format.

    This function extracts key parameters from each section of the config (dataset, general,
    train, model) and logs them using the provided logger.

    Parameters
    ----------
    cfg : Any
        The configuration object, typically an OmegaConf DictConfig, containing dataset,
        general, train, and model sections.
    logger : logging.Logger
        The logger instance to use for logging the configuration.

    Notes
    -----
    - The function relies on `get_nonstatic_cfg_params(cfg)` to extract relevant config entries.
    - No logging configuration (handlers, level) is modified; it uses the provided logger.
    """
    dataset_cfg, general_cfg, train_cfg, model_cfg = get_nonstatic_cfg_params(cfg)
    
    if logger is None:
        logger = logging
    logger.info("Dataset config:\n%s", dataset_cfg)
    logger.info("General config:\n%s", general_cfg)
    logger.info("Training config:\n%s", train_cfg)
    logger.info("Model config:\n%s", model_cfg)

def force_setattr(cfg_section, key, value):
    """Set a key in a DictConfig regardless of whether it exists."""
    struct_mode = OmegaConf.is_struct(cfg_section)
    if struct_mode:
        OmegaConf.set_struct(cfg_section, False)  # temporarily allow new keys
    setattr(cfg_section, key, value)
    if struct_mode:
        OmegaConf.set_struct(cfg_section, True)   # restore struct mode

def safe_setattr(cfg_section, key, value):
    """
    Safely set a value in a DictConfig or normal object.
    Only sets the value if the key already exists (avoiding struct errors).
    """
    if isinstance(cfg_section, omegaconf.DictConfig):
        if key in cfg_section:
            cfg_section[key] = value
    else:
        if hasattr(cfg_section, key):
            setattr(cfg_section, key, value)