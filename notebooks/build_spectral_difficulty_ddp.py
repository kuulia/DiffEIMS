"""Build notebooks/spectral_difficulty_analysis_ddp.ipynb by transforming spectral_difficulty_analysis.ipynb.

The source notebook is still being edited, so instead of a hand-copied port this script
  * replaces the cells that are DDP-specific (intro, setup patches, Section 1 loading),
  * rewrites every other cell with rules (top-10 -> TOP_K, Morgan generator, NN-cache reuse, ...),
  * fails loudly if a rule stops matching or top-10-specific code survives.
Re-run it after the source changes (from notebooks/, any python >= 3.8):  python build_spectral_difficulty_ddp.py
"""
import json, re, sys, argparse

SRC = '/home/ms/diffms/notebooks/spectral_difficulty_analysis.ipynb'
OUT = '/home/ms/diffms/notebooks/spectral_difficulty_analysis_ddp.ipynb'

ap = argparse.ArgumentParser()
ap.add_argument('out', nargs='?', default=OUT)
ap.add_argument('--spec-pkl')   # dry-run overrides
ap.add_argument('--out-dir')
args = ap.parse_args()


def sub1(pattern, repl, s, flags=0, what=''):
    """re.sub that must match exactly once."""
    new, n = re.subn(pattern, repl, s, flags=flags)
    if n != 1:
        raise SystemExit(f'Rule {what or pattern!r} matched {n} times (expected 1) — the source changed, update the rule.')
    return new


# ── DDP-specific cell contents ───────────────────────────────────────────────
INTRO = r'''
# Spectral Feature Difficulty Analysis — DDP runs (`data/new_experiments_ddp`)
Companion to `feature_difficulty_analysis_ddp.ipynb`. That notebook asked which **molecular** features make test
compounds hard to elucidate; this one asks the same question of the **input EI spectra**.

> Generated from `spectral_difficulty_analysis.ipynb` by a build script (rules + DDP-specific cells). Edit the source
> notebook and rebuild rather than editing this one by hand.

Differences from the non-DDP notebook:

* **Kernel.** The prediction pickles are written on LUMI with RDKit 2026.03 / NumPy 2.x — run this notebook in the
  **`diffms_lumi`** env (`envs/build_local_env.sh`).
* **Data layout.** `data/new_experiments_ddp/expN/expN_{pred,true}_r<rank>_<batch>.pkl`, one pickle per rank and
  per-rank batch. Shards are re-interleaved into `DistributedSampler` order and the padding duplicates are dropped
  (see `feature_difficulty_analysis_ddp.ipynb` for details).
* **Outcomes.** 100 candidates per molecule, ranked by occurrence exactly like `K_ACC_Collection.update()`. Each
  molecule stores the rank of the true structure, so top-k hits are available for any k; `TOP_K` in the setup cell
  picks the hit/miss split used throughout (every "top-10" of the original becomes top-`TOP_K`). Top-1 Tanimoto uses
  the top-*ranked* candidate (canonicalised from its InChI), matching `tanimoto_at_1` logged by the test step.
* **Same test set.** All four DDP experiments are evaluated on the same, identically ordered test set (checked in
  Section 1), so the spectrum linkage, spectral and ATMOMACCS features, PCA and spectral clusters are identical across
  experiments; only the outcomes differ.
* The prediction shards only store `(pred_mols, true_mol)`, so each test molecule is linked back to its NEIMS spectrum
  **by structure** (canonical SMILES, falling back to InChIKey for tautomers such as amide ↔ imidic acid that change
  during graph reconstruction). The spectrum pickle is the same as in the non-DDP notebook.
* Hit-dependent figures and CSVs carry `_top{TOP_K}` in their filenames.

'''

SETUP_EXTRA = r'''
# The one knob: every hit/miss split, plot title and hit-dependent output filename uses TOP_K.
# Outcomes are cached as the rank of the true structure, so changing TOP_K needs no reload.
TOP_K = 100
HIT   = f'hit_top{TOP_K}'
HR    = f'hit_rate_top{TOP_K}'
K_REPORT = sorted({1, 10, TOP_K})          # hit columns added to every per-molecule table
'''

SETUP_HELPERS = r'''
# Morgan r=2, 2048 bits — same bits as the legacy Morgan bit-vector call in src/metrics (K_TanimotoSimilarity)
morgan_fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint

def exp_tag(exp_name):
    return exp_name.split(':')[0]

'''

LOAD_MD = r'''
## 1. Load experiment predictions and per-molecule outcomes
Same DDP loading and scoring logic as `feature_difficulty_analysis_ddp.ipynb`: valid candidates → InChI → ranked by
generation count. Candidates are discarded right after scoring (100 candidate Mols × 7.5k molecules × 4 experiments
does not fit in memory); the per-molecule outcomes are cached in `OUT_DIR`. Scoring takes ~10 minutes without the cache.

`true_rank` is the 1-based position of the true InChI in the ranking (`inf` if never generated), so
`hit_top{k} = true_rank <= k`.
'''

LOAD_CODE = r'''
_PRED_RE = re.compile(r'^(?P<prefix>.+)_r(?P<rank>\d+)_(?P<batch>\d+)\.pkl$')

def score_molecule(pred_mols, true_mol):
    """Rank of the true structure among count-ranked valid candidates, and top-1 Tanimoto (tanimoto_at_1)."""
    true_inchi = Chem.MolToInchi(true_mol)
    inchis = [Chem.MolToInchi(mol) for mol in pred_mols if is_valid(mol)]
    ranked = [inchi for inchi, _ in Counter(inchis).most_common()]
    true_rank = ranked.index(true_inchi) + 1 if true_inchi in ranked else np.inf
    tan = 0.0
    if ranked:
        try:
            tan = DataStructs.TanimotoSimilarity(morgan_fp(canonical_mol_from_inchi(ranked[0])), morgan_fp(true_mol))
        except Exception:
            pass
    return true_inchi, true_rank, tan

def load_ddp_outcomes(exp, base=EXP_BASE):
    """Score every rank/batch shard of a DDP run; return per-molecule outcomes in DistributedSampler order."""
    d = os.path.join(base, exp)
    pred_prefix, true_prefix = f'{exp}_pred', f'{exp}_true'
    shards = {}   # rank -> {batch: filename}
    for f in os.listdir(d):
        m = _PRED_RE.match(f)
        if m and m.group('prefix') == pred_prefix:
            shards.setdefault(int(m.group('rank')), {})[int(m.group('batch'))] = f
    if not shards:
        raise FileNotFoundError(f'No {pred_prefix}_r*_*.pkl shards in {d}')
    world = len(shards)
    if sorted(shards) != list(range(world)):
        raise ValueError(f'Ranks are not contiguous 0..{world-1}: {sorted(shards)}')

    t0 = time.time()
    per_rank = {}
    for rank in sorted(shards):
        rows = []
        for b in sorted(shards[rank]):
            pf = shards[rank][b]
            with open(os.path.join(d, pf), 'rb') as fh:
                preds = pkl.load(fh)
            with open(os.path.join(d, pf.replace(pred_prefix, true_prefix, 1)), 'rb') as fh:
                trues = pkl.load(fh)
            if len(preds) != len(trues):
                raise ValueError(f'{pf}: {len(preds)} preds vs {len(trues)} trues')
            for p, t in zip(preds, trues):
                if t is None:   # skipped by the test-step metrics too; kept here so the interleaving stays aligned
                    rows.append({'true_inchi': None})
                    continue
                true_inchi, true_rank, tan = score_molecule(p, t)
                rows.append({'true_smiles': Chem.MolToSmiles(t), 'true_inchikey': Chem.MolToInchiKey(t),
                             'true_inchi': true_inchi, 'true_rank': true_rank, 'tanimoto_top1': tan,
                             'n_generated': len(p)})
            del preds, trues
        per_rank[rank] = rows
        if (rank + 1) % 8 == 0:
            print(f'  {exp}: ranks 0-{rank} scored  ({time.time()-t0:.0f}s)', flush=True)

    # rank r's j-th sample is global position r + world*j
    lens = {len(v) for v in per_rank.values()}
    if len(lens) != 1:
        raise ValueError(f'Ranks hold different numbers of samples: { {r: len(v) for r, v in per_rank.items()} }')
    n_per_rank = lens.pop()
    rows = [None] * (world * n_per_rank)
    for r, rr in per_rank.items():
        for j, row in enumerate(rr):
            rows[r + world * j] = row
    # DistributedSampler pads to a multiple of world by repeating the first pad (< world) indices
    keys, pad = [r['true_inchi'] for r in rows], 0
    for k in range(world - 1, 0, -1):
        if keys[-k:] == keys[:k]:
            pad = k
            break
    odf = pd.DataFrame(rows[:len(rows) - pad])
    odf['global_pos'] = np.arange(len(odf))
    n_none = odf['true_inchi'].isna().sum()
    odf = odf[odf['true_inchi'].notna()].reset_index(drop=True)
    print(f'  {exp}: world={world}  raw={len(rows):,}  -> {len(odf):,} after dropping {pad} padding duplicate(s) '
          f'and {n_none} unparseable true molecule(s)  ({time.time()-t0:.0f}s)')
    return odf

EXP_CONFIGS = {
    'Exp1: Only Gecko':                     dict(exp='exp1'),
    'Exp2: Only Augmented':                 dict(exp='exp2'),
    'Exp3: Augmented + Gecko (sequential)': dict(exp='exp3'),
    'Exp4: Augmented + Gecko (parallel)':   dict(exp='exp4'),
}

OUTCOME_CACHE = f'{OUT_DIR}/outcomes_cache.pkl'
if os.path.exists(OUTCOME_CACHE):
    with open(OUTCOME_CACHE, 'rb') as fh:
        exp_outcomes = pkl.load(fh)
    print(f'Loaded cached outcomes from {OUTCOME_CACHE}')
else:
    exp_outcomes = {exp_name: load_ddp_outcomes(**cfg) for exp_name, cfg in EXP_CONFIGS.items()}
    with open(OUTCOME_CACHE, 'wb') as fh:
        pkl.dump(exp_outcomes, fh)

for exp_name, odf in exp_outcomes.items():
    for k in K_REPORT:
        odf[f'hit_top{k}'] = odf['true_rank'] <= k
    print(f'{exp_name}: n={len(odf):,}  ' + '  '.join(f'top{k}={odf[f"hit_top{k}"].mean():.4f}' for k in K_REPORT)
          + f'  tanimoto_top1_mean={odf.tanimoto_top1.mean():.4f}')

# All experiments should share the same, identically ordered test set
ref_name, ref = next(iter(exp_outcomes.items()))
for exp_name, odf in exp_outcomes.items():
    same = len(odf) == len(ref) and (odf['true_inchi'].values == ref['true_inchi'].values).all()
    print(f'{exp_name}: same test set and order as {exp_tag(ref_name)}: {same}')
'''

NN_CACHE_REUSE = r'''
# The nearest neighbours depend only on the spectra and the split, not on the predictions, so the cache written by
# the non-DDP notebook is reused when present (its length check below still guards against a mismatch)
_nn_cache_nonddp = NN_CACHE.replace(OUT_DIR, 'spectral_difficulty_analysis', 1)
if not os.path.exists(NN_CACHE) and os.path.exists(_nn_cache_nonddp):
    import shutil
    shutil.copy(_nn_cache_nonddp, NN_CACHE)
    print(f'Copied nearest-neighbour cache from {_nn_cache_nonddp}')
'''

FINDINGS_BANNER = r'''
> **These findings are from the non-DDP runs** (`spectral_difficulty_analysis.ipynb`: 10 candidates per molecule,
> top-10 hits, top-1 Tanimoto on the first *sampled* candidate, partly different test sets per experiment). They are
> kept for reference and need to be re-checked against the DDP results above (100 ranked candidates, top-`TOP_K` hits,
> shared test set) before being reused.
'''

# Figures/CSVs whose content depends on the hit definition get _top{TOP_K} in their filename.
HIT_DEPENDENT_OUTPUTS = ['hist_', 'effect_size_heatmap', 'partial_correlations', 'hit_rate_MW_by_Mion',
                         'difficulty_prediction', 'permutation_importance', 'spectral_PCA', 'nn_novelty_ambiguity',
                         'example_spectra', 'spectral_hit_miss', 'spectral_clusters']

STR_LIT = re.compile(r'''(?<![\w'"])(?P<pre>[fFrR]{0,2})(?P<q>['"])(?P<body>(?:(?!(?P=q)).)*?)(?P=q)''')


def rewrite_literal(m):
    pre, q, body = m.group('pre'), m.group('q'), m.group('body')
    if not body:
        return m.group(0)
    new = body
    new = re.sub(r'\bhit10\b', '{_HIT_TAG}', new)
    new = re.sub(r'\btop-10\b', 'top-{TOP_K}', new)
    new = re.sub(r'\bTop-10\b', 'Top-{TOP_K}', new)
    new = re.sub(r'\btop10\b', 'top{TOP_K}', new)
    for stem in HIT_DEPENDENT_OUTPUTS:
        new = re.sub(rf'({re.escape(stem)}[^{q}]*?)\.(png|csv)$', r'\1_top{TOP_K}.\2', new)
    if new == body:
        return m.group(0)
    new = new.replace('{_HIT_TAG}', 'hit{TOP_K}')
    if 'f' not in pre.lower():
        if '{' in body or '}' in body:
            raise SystemExit(f'Cannot turn literal into an f-string (has braces): {m.group(0)!r}')
        pre = 'f' + pre
    return f'{pre}{q}{new}{q}'


def rewrite_code(src):
    s = src
    # cluster summary: per-k hit-rate columns
    s = s.replace("hit_rate_top1=('hit_top1', 'mean'), hit_rate_top10=('hit_top10', 'mean')",
                  "**{f'hit_rate_top{k}': (f'hit_top{k}', 'mean') for k in K_REPORT}")
    s = s.replace("hit10=('hit_top10', 'mean')", "**{f'hit{TOP_K}': (HIT, 'mean')}")
    s = re.sub(r'''(['"])hit_rate_top10\1''', 'HR', s)
    s = re.sub(r'''(['"])hit_top10\1''', 'HIT', s)
    s = re.sub(r'\.hit_top10\b', '[HIT]', s)
    s = STR_LIT.sub(rewrite_literal, s)
    s = re.sub(r'AllChem\.GetMorganFingerprintAsBitVect\(([^,()]+), 2, nBits=2048\)', r'morgan_fp(\1)', s)
    s = s.replace('.applymap(', '.map(')
    # hit-rate colour scale: 0.5 was tuned for top-10 rates
    s = s.replace('vmin=0, vmax=0.5', 'vmin=0, vmax=min(1.0, 2 * max(d[HIT].mean() for d in exp_dfs.values()))')
    return s


# ── Build ────────────────────────────────────────────────────────────────────
src = json.load(open(SRC))
cells_in = [(c['cell_type'], ''.join(c['source'])) for c in src['cells']]
cells_out = []
seen = set()

for kind, s in cells_in:
    if kind == 'markdown':
        if s.startswith('# Spectral Feature Difficulty Analysis'):
            sections = s[s.index('\nSections'):].lstrip('\n')
            s = INTRO.lstrip('\n') + sections
            seen.add('intro')
        elif s.startswith('## 1. Load'):
            s = LOAD_MD.strip('\n'); seen.add('load_md')
        elif s.startswith('## 13. Findings'):
            s = '## 13. Findings\n' + FINDINGS_BANNER + s[len('## 13. Findings'):]
            seen.add('findings')
        else:
            s = re.sub(r'\btop-10\b', 'top-`TOP_K`', s)
            s = re.sub(r'\bTop-10\b', 'Top-`TOP_K`', s)
            if s.startswith('## 9. Spectral space'):
                s += ('\n\nWith the shared DDP test set the PCA and clusters come out identical for every experiment; '
                      'only the hit/miss colouring and hit rates differ.')
        cells_out.append((kind, s))
        continue

    if 'OUTCOME_CACHE' in s:                                    # Section 1 loader
        cells_out.append((kind, LOAD_CODE.strip('\n'))); seen.add('load'); continue

    if 'EXP_BASE' in s and 'SPEC_PKL' in s:                     # setup
        s = sub1(r'^import sys, os, warnings, pickle as pkl$', 'import sys, os, re, time, warnings, pickle as pkl',
                 s, re.M)
        s = sub1(r'Draw, AllChem, DataStructs\b', 'Draw, DataStructs, rdFingerprintGenerator', s)
        s = sub1(r'^from src\.utils import is_valid$', 'from src.utils import is_valid, canonical_mol_from_inchi', s, re.M)
        s = sub1(r"^EXP_BASE\s*=.*$", "EXP_BASE  = 'data/new_experiments_ddp'", s, re.M)
        s = sub1(r"^OUT_DIR\s*=.*$", f"OUT_DIR   = '{args.out_dir or 'spectral_difficulty_analysis_ddp'}'", s, re.M)
        if args.spec_pkl:
            s = sub1(r"^SPEC_PKL\s*=.*$", f"SPEC_PKL  = '{args.spec_pkl}'", s, re.M)
        s = sub1(r'^(os\.makedirs\(OUT_DIR, exist_ok=True\)\n)', r'\1' + SETUP_EXTRA.replace('\\', r'\\'), s, re.M)
        s = sub1(r"^print\('Setup complete\.'\)$",
                 SETUP_HELPERS.lstrip('\n') + "import rdkit\n"
                 "print(f'Setup complete.  rdkit {rdkit.__version__}, numpy {np.__version__}   TOP_K = {TOP_K}')",
                 s, re.M)
        cells_out.append((kind, s)); seen.add('setup'); continue

    s = rewrite_code(s)
    if 'NN_CACHE' in s:                                         # Section 10: reuse the non-DDP NN cache
        s = sub1(r"^(NN_CACHE = f'\{OUT_DIR\}/spectral_nn_cache[^']*'\n)", r'\1' + NN_CACHE_REUSE.lstrip('\n'), s, re.M)
        seen.add('nn')
    if "drop(columns=['true_inchikey'])" in s:
        s = s.replace("drop(columns=['true_inchikey'])", "drop(columns=['true_inchikey', 'true_inchi'])")
    cells_out.append((kind, s))

missing = {'intro', 'load_md', 'load', 'setup', 'nn', 'findings'} - seen
if missing:
    raise SystemExit(f'Source cells not found: {missing} — the source changed, update the builder.')

# Nothing top-10 specific may survive in code (Section 1 is DDP code, the findings are quoted on purpose)
LEFTOVER = re.compile(r"hit_top10|hit_rate_top10|\bhit10\b|\btop-?10\b|\bTop-10\b|AllChem|applymap|"
                      r"data/new_experiments'|dev_pred|dev_resume")
bad = [(i, line) for i, (k, s) in enumerate(cells_out) if k == 'code'
       for line in s.splitlines() if LEFTOVER.search(line)]
if bad:
    raise SystemExit('Top-10 / non-DDP leftovers in code cells:\n' + '\n'.join(f'  cell {i}: {l.strip()}' for i, l in bad))
for i, (k, s) in enumerate(cells_out):
    if k == 'code':
        compile(s, f'<cell {i}>', 'exec')   # syntax check

nb = {'cells': [], 'metadata': src['metadata'], 'nbformat': src['nbformat'], 'nbformat_minor': src['nbformat_minor']}
for kind, s in cells_out:
    c = {'cell_type': kind, 'metadata': {}, 'source': s.splitlines(keepends=True)}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    nb['cells'].append(c)
nb['metadata']['kernelspec'] = {'display_name': 'diffms_lumi', 'language': 'python', 'name': 'diffms_lumi'}
nb['metadata'].setdefault('language_info', {})['version'] = '3.12.3'
with open(args.out, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')
print(f'wrote {args.out}: {len(nb["cells"])} cells from {SRC}')
