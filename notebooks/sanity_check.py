import torch
import sys
from omegaconf import DictConfig
import hydra
sys.path.append('../src')
sys.path

from src.diffusion_model_spec2mol import Spec2MolDenoisingDiffusion
from src.datasets import spec2mol_dataset
from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete

from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.analysis.visualization import MolecularVisualization
from rdkit import RDLogger
from src import utils

@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    RDLogger.DisableLog('rdApp.*')
    datamodule = spec2mol_dataset.Spec2MolDataModule(cfg) # TODO: Add hyper for n_bits
    dataset_infos = spec2mol_dataset.Spec2MolDatasetInfos(datamodule, cfg)
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)

    # We do not evaluate novelty during training
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)
    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features, domain_features=domain_features)
    input_dims = dataset_infos.input_dims
    print(input_dims)
    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features, 'map_location': torch.device('cpu')}
    
    
    model = Spec2MolDenoisingDiffusion.load_from_checkpoint('../../../data/checkpoints/checkpoints/epoch=6.ckpt', **model_kwargs)
    model.eval()
    batch = next(iter(datamodule.val_dataloader()))
    data = batch['graph']

    # 1) Convert to dense
    dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)  # dense_data.X [B, N, Xdim], dense_data.E [B, N, N, Edim]

    dense_data = dense_data.mask(node_mask)  # apply mask to dense data (if needed)

    X, E = dense_data.X, dense_data.E
    y = data.y.float()  # or your target y tensor

    # 2) Optionally apply noise (or identity if just inference)
    # If you want to skip noise, just use identity:
    noisy_data = {'X_t': X, 'E_t': E, 'y_t': y, 'node_mask': node_mask}

    # 3) Compute extra features
    extra_data = extra_features(noisy_data)  # returns utils.PlaceHolder with X, E, y
    print("extra_data.X.shape:", extra_data.X.shape)
    with torch.no_grad():
        out = model(noisy_data, extra_data, node_mask)
    print("Sanity check output:", out)
    
if __name__ == "__main__":
    main()