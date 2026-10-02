"""Slide figures summarising spectral_difficulty_analysis.ipynb.

Reads the CSVs written by the notebook (section 12) and writes three 16:9 PNGs to ./slides/.
Run from notebooks/spectral_difficulty_analysis/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = 'slides'
os.makedirs(OUT, exist_ok=True)

# Reference palette (dataviz skill, light mode) — categorical slots in fixed order
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#e1e0d9', '#c3c2b7'
FIGSIZE, DPI = (11, 6.2), 200

EXPS = {'Exp1': 'Exp1 · GECKO only',
        'Exp2': 'Exp2 · Augmented only',
        'Exp3': 'Exp3 · Augmented + GECKO (sequential)',
        'Exp4': 'Exp4 · Augmented + GECKO (parallel)'}
EXP_COLORS = dict(zip(EXPS, [BLUE, ORANGE, AQUA, YELLOW]))
EXPS_SHORT = {'Exp1': 'GECKO only', 'Exp2': 'Augmented only',
              'Exp3': 'Augmented + GECKO\n(sequential)', 'Exp4': 'Augmented + GECKO\n(parallel)'}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'font.size': 15, 'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.edgecolor': AXIS,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
})


def titles(fig, title, subtitle):
    fig.text(0.04, 0.95, title, fontsize=19, fontweight='bold', color=INK, va='top')
    fig.text(0.04, 0.885, subtitle, fontsize=14, color=INK2, va='top')


def footnote(fig, text):
    fig.text(0.04, 0.025, text, fontsize=11, color=MUTED, va='bottom')


# ---------------------------------------------------------------------------
# Figure 1 — strongest spectral correlates of success, before/after size control
# ---------------------------------------------------------------------------
FEATURES = {  # the four features with the largest mean |ρ| that are consistent in sign
    'nPeaks': 'Number of peaks',
    'Entropy': 'Spectral entropy',
    'Top5_inten_frac': 'Share of intensity in top 5 peaks',
    'LowMz_inten_frac': 'Share of intensity below m/z 50',
}
part = pd.read_csv('results_partial_correlations.csv')
part = part[(part.Outcome == 'hit_top10') & part.Feature.isin(FEATURES)]
agg = part.groupby('Feature')[['rho_raw', 'rho_partial']].mean().loc[list(FEATURES)[::-1]]
shrink = 1 - agg.rho_partial.abs().sum() / agg.rho_raw.abs().sum()

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
fig.subplots_adjust(left=0.33, right=0.95, top=0.72, bottom=0.17)
y = np.arange(len(agg))
ax.axvline(0, color=AXIS, lw=1, zorder=0)
for yi, (raw, par) in zip(y, agg.values):
    ax.annotate('', xy=(par, yi), xytext=(raw, yi),
                arrowprops=dict(arrowstyle='-|>', color=AXIS, lw=2, shrinkA=7, shrinkB=7), zorder=1)
ax.scatter(agg.rho_raw, y, s=140, facecolor='white', edgecolor=MUTED, lw=2, zorder=2, label='Before size correction')
ax.scatter(agg.rho_partial, y, s=140, color=BLUE, edgecolor='white', lw=2, zorder=3,
           label='After size correction')
ax.set_yticks(y, [FEATURES[f] for f in agg.index], fontsize=15, color=INK)
ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)
ax.set_xlim(-0.25, 0.25)
ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
ax.set_xlabel('Correlation with a top-10 hit (Spearman ρ)')
ax.xaxis.grid(True, color=GRID, lw=1)
ax.set_axisbelow(True)
ax.set_ylim(-0.9, len(agg) - 0.5)
ax.text(-0.245, -0.7, '← more common in misses', fontsize=12, color=MUTED, va='center')
ax.text(0.245, -0.7, 'more common in hits →', fontsize=12, color=MUTED, va='center', ha='right')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.07), ncol=2, frameon=False, fontsize=13,
          handletextpad=0.3, columnspacing=1.5)
titles(fig, 'Spectral complexity mostly reflects molecule size',
       f'The correlations drop by about {shrink:.0%} once molecular size is taken into account')
footnote(fig, 'Averaged over the four models (about 7,500 test spectra each). Size correction: partial Spearman on MW and heavy-atom count.')
fig.savefig(f'{OUT}/fig1_spectral_features_vs_size.png', dpi=DPI)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2 — effect of the molecular-ion peak depends on training data
# ---------------------------------------------------------------------------
M_BINS, M_LABELS = [-0.01, 1, 10, 100.01], ['Absent\n(< 1 %)', 'Weak\n(1-10 %)', 'Strong\n(≥ 10 %)']
rates = {}
for e in EXPS:
    d = pd.read_csv(f'results_{e}_per_molecule.csv')
    rates[e] = 100 * d.groupby(pd.cut(d.M_rel_inten, M_BINS, labels=M_LABELS), observed=True)['hit_top10'].mean()

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
fig.subplots_adjust(left=0.10, right=0.80, top=0.72, bottom=0.22)
x = np.arange(len(M_LABELS))
for e, r in rates.items():
    ax.plot(x, r.values, color=EXP_COLORS[e], lw=2.5, solid_capstyle='round', zorder=2, label=EXPS[e])
    ax.scatter(x, r.values, s=90, color=EXP_COLORS[e], edgecolor='white', lw=2, zorder=3)
# Direct labels at the right end, nudged only to avoid overlap
ends = sorted(((r.values[-1], e) for e, r in rates.items()))
placed = []
for v, e in ends:
    yv = max(v, placed[-1] + 0.9) if placed else v
    placed.append(yv)
    ax.text(x[-1] + 0.12, yv, f'{e}  {rates[e].values[-1]:.0f} %', va='center', fontsize=13, color=INK)
ax.set_xticks(x, M_LABELS, fontsize=14)
ax.set_xlim(-0.2, x[-1] + 0.1)
ax.set_ylim(0, 17)
ax.set_yticks([0, 5, 10, 15])
ax.set_ylabel('Top-10 hit rate (%)')
ax.set_xlabel('Molecular-ion (M⁺·) intensity, % of base peak')
ax.yaxis.grid(True, color=GRID, lw=1)
ax.set_axisbelow(True)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.04), ncol=2, frameon=False, fontsize=12,
          handlelength=1.5, columnspacing=1.2)
titles(fig, 'M⁺· helps some models but not others',
       'Exp2 does better when M⁺· is strong; Exp1 and Exp3 do better when it is absent')
footnote(fig, 'Spectra per group: 2,852 absent, 847 weak, 3,789 strong (Exp2 to Exp4). The Exp2 and Exp3 trends remain after size correction.')
fig.savefig(f'{OUT}/fig2_molecular_ion_by_training.png', dpi=DPI)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3 — spectral features add no predictive power beyond structure
# ---------------------------------------------------------------------------
ml = pd.read_csv('results_difficulty_prediction.csv')
auc = ml.pivot(index='Experiment', columns='Features', values='ROC-AUC (hit10)').loc[list(EXPS)]
SETS = [('molecular', 'Molecular features', BLUE),
        ('spectral', 'Spectral features', ORANGE),
        ('combined', 'Both combined', AQUA)]

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
fig.subplots_adjust(left=0.10, right=0.97, top=0.72, bottom=0.21)
x = np.arange(len(auc))
w = 0.22
for k, (col, label, c) in enumerate(SETS):
    xs = x + (k - 1) * (w + 0.02)
    ax.bar(xs, auc[col] - 0.5, bottom=0.5, width=w, color=c, label=label, zorder=2)
    for xi, v in zip(xs, auc[col]):
        ax.text(xi, v + 0.008, f'{v:.2f}', ha='center', va='bottom', fontsize=12, color=INK2)
ax.set_xticks(x, [f'{e}\n{EXPS_SHORT[e]}' for e in auc.index], fontsize=13)
ax.tick_params(axis='x', length=0)
ax.set_ylim(0.5, 1.0)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_ylabel('ROC-AUC for predicting a top-10 hit\n(0.5 = random guess)')
ax.yaxis.grid(True, color=GRID, lw=1)
ax.set_axisbelow(True)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.04), ncol=3, frameon=False, fontsize=13)
titles(fig, 'Spectral features do not improve difficulty prediction',
       'Molecular features alone do as well as both combined; spectral features alone do worse')
footnote(fig, 'Gradient-boosted trees, 5-fold cross-validation. Molecular: 17 RDKit descriptors + 175 ATMOMACCS keys. Spectral: 17 features.')
fig.savefig(f'{OUT}/fig3_prediction_molecular_vs_spectral.png', dpi=DPI)
plt.close(fig)

print('Wrote', sorted(os.listdir(OUT)))
