# Encoder pretraining on 3.3M TMS NEIMS spectra

Status (2026-11-09): planned only, nothing built.

## Goal

Pretrain the encoder on NEIMS spectra for the TMS-derivatized form of the whole guarded
`combined` pool (about 3.3M molecules). Then fine-tune end-to-end on the same data as the
article. Fine-tuning data and the existing in-RAM loader stay as they are for that stage.

## Decisions

| Topic | Decision |
|---|---|
| Spectra | TMS only ("probably", confirm before paying for NEIMS) |
| Nodes | Free choice. Plan uses 4 nodes = 32 GCD ranks = 32 shards, matching `general_encoder.yaml` |
| LR | Article's 2e-4 is believed to be a single-GPU base value, so keep `lr: 0.0002` in `train_encoder.yaml`. Code multiplies by sqrt(GCDs): peak 1.13e-3 on 32 GCDs. See open item 1 |
| Loader | One shard per rank instead of rewriting the loader |

Numbers at 4 nodes (before filtering): 3,157,320 train / 32 = about 98.7k per shard, roughly
8 GB per rank. Batch 128 per GCD = 4,096 per step, 771 steps per epoch, 77,100 steps in 100
epochs. The memory figure scales a code comment in `spec2mol_dataset.py` (about 40 GB per rank
on a larger dataset), not a measurement.

Why shards: the current `_build_splits` holds everything in RAM per rank, about 280 GB per rank
at 3.3M, or 2.3 TB per node against about 512 GB. Shards also fix `parse_mgf` holding everything
in RAM and the lack of mid-step resume in NEIMS preprocessing, because each shard is a separate
array task.

## Steps

Input must be the rebuilt, guarded CSVs on LUMI: `data/fp2mol/combined/preprocessed/combined_{train,val}.csv`.
The local copies are older and still contain guarded molecules.

1. **Clean SMILES.** Convert InChI to 2D SMILES and normalize tautomers, so amides are not
   written as iminols (`N=C(O)`). Keep a `split` column. Do this before step 2, or the iminol
   OH gets silylated into a wrong structure. `combined_inchi_to_smiles.py` does neither yet.
2. **Derivatize.** Replace active hydrogens on hydroxyl, carboxyl, thiol and amine groups with
   TMS. Molecules with no such group stay as they are (65% of `mixed_augment_tms` contains Si).
   The rule-based code is NOT in `diffms`, `neims-pipeline` or `mist`: the article has a TODO
   saying Hilda can explain it, so it needs to be obtained or rewritten to match.
3. **Guard the TMS structures.** Apply the connectivity-key guard to the final structures, using
   AtmoMACCS, AtmoMACCS TMS and the annotated Yee set (`reference_inchi.csv`, including InChIs in
   its notes). Required, not optional: the guard only checked underivatized structures, and
   81 non-TMS parents of annotated Yee derivatives (98 rows of `combined_train`) would derivatize
   into exact Yee targets. Also exclude the test/val molecules of the TMS spectral datasets,
   re-apply `_passes_filter`, set a 100 heavy-atom limit (each TMS adds 4), dedupe, and cut val to
   about 10k (170k now).
4. **NEIMS.** Simulate spectra for the step 3 structures (external step, needs the spectra).
5. **Shard.** `neims_output_to_mgf`, then randomly assign to 32 shards. Each shard gets its own
   `labels.tsv`, split file, `df_*.pkl` and `*_subforms*.pkl`.
6. **Preprocess.** Run `neims_preprocessing` per shard as a Slurm array on LUMI-C, with
   `split_mode: predefined`.
7. **Equalize.** After the last step that can drop spectra, trim every shard to the same train
   and the same val count, or DDP hangs. Log what was trimmed.
8. **Stats.** Compute atom, edge, node and valency stats once on CPU from the final TMS labels.
   Do NOT reuse `data/fp2mol/combined/stats`: TMS changes the molecules. Add a
   `prepare_data_only` path to `spec2mol_main.py`, as in `fp2mol_main.py`.
9. **Config.** New `configs/dataset/<name>.yaml` with per-rank paths via `${oc.env:RANK,0}` for
   labels, split, spectra and subforms. Keep `datadir` shared, and check that `_stat_root` and
   `_sentinel_path` stay shared across ranks. Set `max_heavy_atoms: 100`.
10. **Sampler.** Add `train.use_distributed_sampler` to `spec2mol_main.py` (default true, false
    for this run), since each rank already has its own data. Keep `skip_test: True`.
11. **Smoke test.** `dev-g`, 30 minutes, `train.limit_train_batches` small. Check shard sizes,
    `Scaled LR` and per-rank RSS before the full run.
12. **Train.** `python src/spec2mol_main.py --config-name=config_encoder`. Encoder-only training
    skips the decoder (`lambda_train: [0, 0, 1]`), so it can run alongside decoder pretraining.
13. **Fine-tuning data.** Add the Yee guard to `build_guarded_splits.py` (`LEAKAGE_GUARD_DATASETS`
    only covers AtmoMACCS), regenerate `split_random_guarded.tsv` and the preprocessed CSVs.
14. **Fine-tune.** As in the article, with `general.encoder` set to the new encoder and
    `general.decoder` set to the new guarded decoder. Update the article's methods.

## Known trade-offs

- Each rank keeps its 1/32 of the data for the whole run. Every molecule is still seen once per
  epoch, but batches only mix within a shard.
- Shard count equals rank count, so changing the node count means re-sharding.

## Open items

1. Confirm the encoder LR convention before launching: check `.hydra/config.yaml` and the
   `Scaled LR` log line of the run that made the current encoder. e2e notes in `train_e2e.yaml`
   record divergence at a 1.13e-3 peak, there before gradient clipping was added.
2. Decoder LR uses the opposite reading. `train_decoder_pretrain.yaml` has base 6.25e-5, a true
   5e-4 peak on 64 GCDs. If 5e-4 is a single-GPU base, the equivalent peak is 4e-3 (base 5e-4).
   Compare validation loss with the old pretraining before changing it.
3. Check whether the existing TMS datasets contain silylated iminols from the same tautomer issue.
4. Which accounts to bill. On 2026-11-09 `project_462001155` was out of CPU hours and the decoder
   GPU job was switched to `project_462001448`.
