# ==== analysis helpers: clustered statistics, figures ====
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

def subject_level(df, metric, by=('method', 'cond')):
    """Average repeated runs (seeds) within subject -> one observation per subject (the independent unit)."""
    return df.groupby(list(by) + ['victim'])[metric].mean().reset_index()

def boot_ci_mean(x, n_boot=10000, seed=0, alpha=0.05):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return (np.nan, np.nan)
    r = np.random.default_rng(seed); idx = r.integers(0, len(x), (n_boot, len(x)))
    m = x[idx].mean(1); return tuple(np.quantile(m, [alpha / 2, 1 - alpha / 2]))

def summarize(df, metrics, by=('cond', 'method')):
    rows = []
    for key, g in df.groupby(list(by)):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        for m in metrics:
            s = g.groupby('victim')[m].mean()
            lo, hi = boot_ci_mean(s.values)
            row[m] = s.mean(); row[m + '_lo'] = lo; row[m + '_hi'] = hi
        row['n_subjects'] = g.victim.nunique(); row['n_runs'] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)

def paired_test(a, b, label, alternative='two-sided'):
    """a, b: Series indexed by subject (already averaged over seeds). Tests median(a-b) = 0."""
    idx = a.index.intersection(b.index); d = (a.loc[idx] - b.loc[idx]).astype(float).dropna()
    d = d.round(10)                      # exact ties identified after rounding to 1e-10 (floating-point noise is not a difference)
    lo, hi = boot_ci_mean(d.values)
    nz = d[d != 0]
    if len(nz) < 5: p, stat = np.nan, np.nan
    else:
        res = wilcoxon(nz.values, alternative=alternative, zero_method='wilcox'); p, stat = res.pvalue, res.statistic
    r_pos = (nz > 0).sum(); r_neg = (nz < 0).sum()
    ranks = pd.Series(np.abs(nz.values)).rank().values
    rbc = (ranks[nz.values > 0].sum() - ranks[nz.values < 0].sum()) / ranks.sum() if len(nz) else np.nan
    return dict(comparison=label, n_subjects=len(d), mean_diff=d.mean(), ci_lo=lo, ci_hi=hi,
                median_diff=d.median(), n_pos=int(r_pos), n_neg=int(r_neg), n_zero=int((d == 0).sum()),
                wilcoxon_W=stat, p_value=p, rank_biserial=rbc)

def holm(pvals):
    p = np.asarray(pvals, float); out = np.full_like(p, np.nan); ok = ~np.isnan(p)
    order = np.argsort(p[ok]); m = ok.sum(); adj = np.empty(m); prev = 0
    for k, i in enumerate(order):
        prev = max(prev, min(1.0, (m - k) * p[ok][i])); adj[i] = prev
    out[ok] = adj; return out
