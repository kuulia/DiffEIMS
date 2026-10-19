# Global Topological Features for DiffMS — Design Plan

Author: Linus (drafted with Claude, 2026-08-04)

## 1. The problem

DiffMS runs in two stages. The MIST-style spectra encoder (`SpectraEncoderGrowing`,
`src/mist/models/spectra_encoder.py`) reads an EI-MS spectrum and, via a hierarchical
`FPGrowingModule`, predicts a Morgan fingerprint (`morgan_nbits`, currently 2048 or 4096
bits). That predicted vector becomes `data.y`, the global conditioning signal handed to
the graph-transformer discrete diffusion decoder (`GraphTransformer(V2)`,
`src/models/transformer_model.py`) at every denoising step
(`diffusion_model_spec2mol.py`, `merge == "mist_fp"` path). Separately, at each
denoising step the decoder also receives handcrafted features computed from its *own
current, partially-denoised graph* — cycle counts and Laplacian eigenfeatures
(`src/diffusion/extra_features.py`) and charge/valency/weight
(`src/diffusion/extra_features_molecular.py`). Those are self-conditioning signals, not
spectrum-derived evidence.

Morgan fingerprints only encode local circular neighborhoods (radius `r`, typically
2–3 bonds). Two molecules that differ only far apart — e.g. the same substituent
hanging off a long alkyl chain vs. off a macrocycle, or a substituent sitting on one
ring of a fused/bridged/spiro system vs. two separate rings — can produce nearly
identical Morgan bit patterns. So the decoder's only spectrum-derived evidence
under-determines large-scale shape: ring size, how many fused/nested rings exist,
branching depth, chain length. It has no direct way to know "this should end up
macrocyclic" vs "this should end up as a long branching chain" from `data.y` alone.

## 2. Two places to inject global topology

**Point A — encoder-predicted descriptor (spectrum → topology).** Add a second small
prediction head to `SpectraEncoderGrowing`, parallel to the existing
`spectra_predictor` (`FPGrowingModule`), reading from the same pooled representation
(`encoder_output` / `aux_outputs["h0"]`). Train it against a real RDKit-computed label
from the true molecule, the same way the Morgan target is trained today. Concatenate
its output onto `data.y` so it augments the vector the decoder is conditioned on at
every step. This is new information the model doesn't currently have.

**Point B — self-conditioning descriptor (current graph state → topology).** Alongside
the existing cycle/eigenvalue features in `extra_features.py`, compute genuinely global
descriptors (ring-size histogram, diameter, Wiener index) of the *current noisy graph*
`X_t, E_t` at each reverse-diffusion step, so the decoder gets live topological
feedback on what it has generated so far, not just what the spectrum implies about the
final answer. This complements Point A rather than duplicating it: A says what the
target should look like; B says what the model currently has.

Recommendation: build Point A first — it's the literal reading of "predict from the
spectra" and is where the missing information actually is. Point B is a cheaper,
lower-risk follow-up.

## 3. Candidate feature set

**Tier 1 — new, cheap, targets the stated ring/branching-blindness problem directly:**

- Ring-size histogram: counts of SSSR rings of size 3, 4, 5, 6, 7, 8+ (richer than
  MACCS's binary "ring of size N present" keys)
- Number of separate fused-ring *systems* vs. total ring count — distinguishes one
  fused polycyclic cluster from several independent small rings
- Aromatic vs. non-aromatic ring counts
- Spiro-atom count, bridgehead-atom count (`rdMolDescriptors.CalcNumSpiroAtoms`,
  `CalcNumBridgeheadAtoms`) — directly detects nested/bridged topology
- Longest acyclic chain length; overall graph diameter and radius
  (`rdmolops.GetDistanceMatrix`)
- Branching index: Bertz complexity (`GraphDescriptors.BertzCT`), or degree-variance /
  count of tertiary+quaternary carbons
- Wiener index (sum of all pairwise shortest-path distances) — the classical global
  topological invariant, and cheap to compute

**Tier 2 — reuse fingerprint machinery already in the repo:**

- Full 167-bit MACCS keys — already implemented (`_get_maccs` in
  `src/mist/data/featurizers.py`, and `pyGenMACCSKeys` in `src/ATMOMACCS.py`). Several
  keys are exactly ring-size/ring-count/branching questions (11, 19, 22, 83, 96, 98,
  101, 105, 112, 145, 163, 165) — cheapest possible first experiment since the code
  already exists and is wired into `FingerprintFeaturizer`.
- ATMO fragment keys — recommend *not* pulling in the full ~156-bit ATMO block for a
  general-purpose model. Many of those keys are atmospheric-oxidation-specific
  (nitrate esters, peroxynitrates, hydroperoxides) and are only meaningfully populated
  on the GECKO/atmospheric dataset variants (`atmomaccs`, `gecko_new_atmomaccs`, etc. —
  note these configs currently still use plain `morgan_nbits: 2048`, ATMOMACCS isn't
  actually wired into y-conditioning anywhere yet). Worth revisiting per-dataset rather
  than as a universal default.

**Tier 3 — optional stretch, still cheap RDKit one-liners:** Balaban J index,
eccentric connectivity index, Kier shape indices (kappa1–3) — these specifically
capture "elongated chain vs. compact ring cluster," which is the exact axis the
problem statement is about.

**Recommended starting scope:** Tier 1 (~15–20 scalars) + MACCS-167 ≈ 180–190 extra
dims. Skip full ATMO block initially.

## 4. Labeling / feature-extraction pipeline

- New module `src/topo_features.py`: `compute_topo_features(mol) -> np.ndarray`
  wrapping the Tier-1 RDKit calls, plus a manifest describing each dimension's type
  (binary / count / continuous) so downstream code knows how to normalize it and which
  loss term applies.
- Extend `FingerprintFeaturizer` (`src/mist/data/featurizers.py`): add `fp_name`
  entries `"maccs"` (already present) and new `"topo"`, and update the
  `get_fingerprint_size` classmethod's `fp_name_to_bits` table.
- Add these to the `fp_names` list passed into `FingerprintFeaturizer(...)` in
  `_build_splits` / `compute_dataset_stats` (`src/datasets/spec2mol_dataset.py`),
  alongside the existing `"morgan4096"` — this reuses the existing plumbing exactly.

## 5. Threading the new dimensions through the model

- `_compute_static_dims` (`src/datasets/spec2mol_dataset.py`) currently derives
  `input_dims["y"]` / `output_dims["y"]` solely from `morgan_nbits`. Add a new config
  key `cfg.dataset.topo_nbits` and fold it in:
  `output_dims["y"] = morgan_nbits + topo_nbits` (same for input, plus the existing
  `+1` time conditioning and extra-feature dims).
- `GraphTransformerV2`'s `y_encoder_fp_dim` (currently `cfg.dataset.morgan_nbits`,
  passed in `diffusion_model_spec2mol.py`) needs to become `morgan_nbits + topo_nbits`.
  **To verify before implementing:** exactly how `transformer_model.py` consumes
  `y_encoder_fp_dim` internally — haven't traced that file yet.
- `SpectraEncoderGrowing`: add a second head, e.g.
  `self.topo_predictor = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, topo_nbits))`,
  reading `encoder_output`. Store as `aux_outputs["topo_preds"]`.
- `diffusion_model_spec2mol.py`: everywhere `data.y = aux["int_preds"][-1]` (the
  `merge == "mist_fp"` branch, in `training_step`/`validation_step`/`test_step`),
  extend to `data.y = torch.cat([aux["int_preds"][-1], aux["topo_preds"]], dim=-1)`.
  The ground-truth `data.y` built in `GraphFeaturizer._featurize`
  (`src/mist/data/featurizers.py`) needs the matching extension so encoder output and
  training target have the same shape.

## 6. Loss changes

- Split the fingerprint loss into a BCE term (Morgan bits + MACCS bits, all binary)
  and an MSE/Smooth-L1 term for continuous counts (ring counts, Wiener index — likely
  `log1p`-normalized). Add a configurable weight for the new term, following the
  existing `cfg.model.lambda_train` convention in `TrainLossDiscrete`
  (`src/metrics/train_metrics.py` — **to verify**: exact current loss composition
  there before splitting it).
- Report per-segment metrics separately (Morgan-BCE-acc, MACCS-BCE-acc, topo-MSE)
  rather than one blended `y_CE`, for debuggability — the existing `val_y_CE` /
  `test_CE` assume one homogeneous space.
- Consider following `ATMOMACCS.py`'s own convention (`_bespe`): turn counts into
  binary threshold questions ("num rings > 1", "diameter > 8") rather than raw
  regression. That file was clearly designed around binary questions being easier to
  train reliably than regression for these kinds of molecular counts — worth reusing
  that judgment rather than re-deriving it.

## 7. Evaluation plan

1. Baseline: current Morgan-only conditioning (existing checkpoints).
2. Variant A: + MACCS-167 only (binary) — cheapest, reuses existing code entirely.
3. Variant B: + MACCS-167 + Tier-1 custom scalars.
4. Variant C (optional, atmospheric datasets only): + trimmed ATMO block.

Metrics: existing `K_ACC_Collection` / `K_SimilarityCollection` (top-k accuracy,
Tanimoto), **plus** a stratified test-set split into "long-chain / low-ring-count"
vs. "polycyclic / fused / nested-ring" molecules (bucket using the same Tier-1
features computed on ground truth), reporting top-k/Tanimoto separately per bucket.
This directly tests whether the change fixes the specific failure mode motivating it,
rather than relying on an average metric that could hide a wash between buckets.

**Cheap sanity check before touching the decoder at all:** verify the encoder can
predict the new targets from spectra in the first place — report standalone
BCE-accuracy/MSE for `topo_preds` vs. ground truth on the validation set. If EI-MS
fragmentation genuinely carries no signal for, say, Wiener index, no amount of decoder
wiring downstream will help, and that's worth knowing early.

## 8. Risks / open questions

- **Redundancy with Point-B-style features already in the model.** The existing
  `ExtraFeatures` (cycle counts, eigenvalues) are computed from the decoder's own
  in-progress graph, not predicted from the spectrum — different role, likely still
  complementary, but worth checking overlap empirically rather than assuming.
- **Signal-to-noise from the encoder side.** EI-MS fragmentation is often dominated by
  local bond-weakness effects; whether global ring topology is actually recoverable
  from a spectrum is an open empirical question, addressed by the Phase-0/Section-7
  sanity check below before investing in decoder wiring.
- **Dimensionality:** +~190 y-dims is modest relative to the existing 2048/4096-bit
  Morgan vector; compute cost increase should be minor.
- Two implementation details flagged above as unverified and needing a read before
  coding: `y_encoder_fp_dim` consumption in `transformer_model.py`, and the current
  loss composition in `train_metrics.py` / `TrainLossDiscrete`.

## 9. Phased implementation checklist

- **Phase 0 (no model changes):** implement `topo_features.py`, compute Tier-1 +
  MACCS on existing training molecules, confirm the "long-chain vs. nested-ring"
  stratification actually shows a measurable accuracy gap in the *current* model.
  Confirms the problem before investing further.
- **Phase 1:** wire ground-truth topo features into `FingerprintFeaturizer` /
  `GraphFeaturizer` as a new dataset stat; extend `_compute_static_dims`.
- **Phase 2:** add the encoder's auxiliary head + loss; verify standalone predictive
  accuracy for topo features from spectra alone.
- **Phase 3:** wire the predicted topo vector into decoder conditioning
  (`merge == "mist_fp"` path); update `y_encoder_fp_dim` in `GraphTransformerV2`.
- **Phase 4:** run the Variant A/B/C ablation with stratified metrics.
- **Phase 5 (optional):** add Point-B self-conditioning topo features to
  `extra_features.py` alongside the existing cycle/eigen features.

## 10. Staged, compute-aware experiment ordering (addendum)

Full decoder pretraining runs on the "combined" fp2mol corpus
(`FP2MolDenoisingDiffusion`, `src/fp2mol_main.py`, `configs/dataset/fp2mol.yaml`) —
this is the "millions of molecules" cost. The encoder trains on the much smaller
paired spectra-molecule datasets. That cost asymmetry should drive the order of
experiments, and existing checkpoints in `data/checkpoints/checkpoints/`
(`decoder.ckpt`, `encoder_epoch=45.ckpt`, plus the full joint `diffms_canopus.ckpt` /
`diffms_msg.ckpt`) mean neither "pretrain the decoder with all the features" nor
"pretrain the encoder to see what's predictable" needs to start from scratch.

**Step 1 (near-free) — probe the existing encoder checkpoint.** Freeze
`encoder_epoch=45.ckpt`, train only a small linear/MLP probe on its pooled
representation (`aux_outputs["h0"]`) to predict the Tier-1 + MACCS topo features, on
the existing paired dataset. This directly answers "which features can be reliably
predicted from EI-MS" without training anything expensive. If a feature can't beat a
trivial baseline here, deprioritize it — no amount of decoder work will make a
non-existent signal usable.

**Step 2 (medium cost) — ceiling-test the decoder via warm-start, not full pretrain.**
Load `decoder.ckpt`, widen `mlp_in_y` to the new dims, and apply the `"ft-input"`
strategy already implemented in `apply_decoder_finetuning`
(`src/spec2mol_encoder.py`) to freeze the pretrained transformer body and train only
the input projection on ground-truth (not predicted) Morgan+MACCS+topo features. This
approximates "assume a perfect encoder" at a fraction of the cost of a full pretrain,
since only a thin layer is being trained and it can run on a subsample of the combined
corpus rather than the full set.

  Practical blocker to fix first: `load_weights` (`src/spec2mol_encoder.py`) currently
  builds `filtered_state_dict` by key-name match only (`k in model_state_dict`) and
  then calls `load_state_dict(..., strict=False)`. `strict=False` suppresses
  missing/unexpected *keys*, not shape mismatches on matched keys — loading a
  checkpoint into a resized `mlp_in_y` will still raise. Needs a small patch: skip (or
  partially copy, zero/random-init the new columns of) any key whose shape doesn't
  match before calling `load_state_dict`.

**Step 3 (expensive, gated on 1 and 2) — full pretrain from scratch.** Only commit to
this if the probe shows the features are predictable *and* the warm-started fine-tune
shows the decoder benefits from them with a frozen body. A full joint retrain might
still beat the frozen-body version if the new features need deeper architectural
adaptation than a single input projection allows, but at that point the spend is
backed by evidence from Steps 1–2 instead of speculation.

This effectively reorders Phases 1–4 above: run Phase 1 (labeling) as before, then do
Step 1 (encoder probe) in place of a full Phase 2, then Step 2 (warm-started decoder
fine-tune) in place of a full Phase 3, and only fall back to genuine full-scale
retraining (original Phases 2–3) if the cheap versions are inconclusive.
