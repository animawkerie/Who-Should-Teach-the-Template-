# ==== pacau_rev: revision experiments (reviewer-driven extensions of pacau_core) ====
# Every extension is OFF by default: with default settings run_stream2() consumes exactly the same random
# numbers as pacau_core.run_stream() and reproduces its outputs bit-for-bit (asserted in the notebook).
import numpy as np, pandas as pd, time, os, multiprocessing as mp
from pacau_core import (Authenticator, Accountant, METHODS, CONDITIONS, ALPHA_GRID, DEFAULT_CFG, sigmoid, add_bias,
                        eer, auc, partition, build_run, attack_plan, sid, cmu_transform)

# ------------------------------------------------------------------ policies
METHODS2 = dict(METHODS)
# single-factor contrast completions (Reviewer 2): P vs B2p isolates the silent gate under DP with identical
# budget + pacing; P vs B3p isolates DP noise/clipping with an identical (accountant-driven) update schedule.
METHODS2['B2p_DP_selfupdate_paced'] = dict(METHODS['B2_DP_selfupdate'], pacing=True)
METHODS2['B3p_gate_sched'] = dict(METHODS['P_gate_DP_budget'], dp=False, sched=True)
# selection-effect control for labels-only updating: also challenge a random 5% of silently accepted samples
METHODS2['B1c_audit5'] = dict(METHODS['B1c_labels_only'], audit=0.05)
# grace-window labelling: after a passed step-up the next G presented samples are treated as verified
for k in ['B1c_labels_only', 'B2c_DP_labels_only', 'P_gate_DP_budget']:
    METHODS2[k + '_grace'] = dict(METHODS[k], grace=True)

CORE = ['B0_static', 'B1_selfupdate', 'B1b_double_thr', 'B1c_labels_only', 'B2_DP_selfupdate', 'B2p_DP_selfupdate_paced',
        'B2c_DP_labels_only', 'B3_gate_nonDP', 'B3p_gate_sched', 'P_gate_DP_budget']
KEY = ['B0_static', 'B1_selfupdate', 'B1c_labels_only', 'B2c_DP_labels_only', 'B3_gate_nonDP', 'P_gate_DP_budget']

# ------------------------------------------------------------------ attack conditions
CONDITIONS2 = dict(CONDITIONS)
CONDITIONS2.update({
    'C3n_oracle_motornoise_20': ('oracle_noisy', 0.20),   # C3 + human motor noise: physically realisable on a keyboard
    'C5_gateaware_20':          ('gate_aware', 0.20),     # white-box incl. novelty statistic and trust radius R
    'C6_generative_20':         ('generative', 0.20),     # samples from a generative model of the victim fitted on leaks
    'C3s_oracle_strong_40':     ('oracle', 0.40),         # stronger dose: 40% of stream, bursts twice as long (20 on CMU)
    'H_handover_sessionstart':  ('handover', 0.0),        # owner signs in (strong factor) then hands the device over: one natural burst per session
})
FAR_GRID = np.logspace(-3, 0, 25)          # DET grid (probe FAR) on which per-run FRR is stored

# ------------------------------------------------------------------ authenticator with operating-point revalidation
class Authenticator2(Authenticator):
    """Adds: (i) FAR-targeted threshold revalidation after every (re)fit on the public reference negatives,
    (ii) accountant-driven scheduling without noise (B3p), (iii) accountant composition across released models."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.tau_acc, self.tau_upd = self.c['tau_acc'], self.c['tau_upd']
    def fit(self, steps):
        super().fit(steps)
        if self.c.get('far_target'):
            neg = self.score(self.Xcal[self.ycal == 0])                 # public reference set only (bg_cal)
            ft = self.c['far_target']
            self.tau_acc = float(np.quantile(neg, 1 - ft))
            self.tau_upd = float(max(self.c['tau_upd'], np.quantile(neg, 1 - ft / 4)))
    def can_spend(self, t, horizon):
        if not ((self.m['dp'] or self.m.get('sched')) and self.m['budget']): return True
        if self.acct.eps(self.steps + self.c['upd_steps']) > self.c['eps_budget']: return False
        if self.m['pacing']:
            if not hasattr(self, 'min_gap'): self.min_gap = horizon / max(self.n_affordable(), 1)
            return (t - self.last_update_t) >= self.min_gap
        return True
    def admit(self, X, p, verified):
        c0 = self.c; self.c = dict(c0, tau_acc=self.tau_acc, tau_upd=self.tau_upd)
        try: return super().admit(X, p, verified)
        finally: self.c = c0

class ComposedAccountant(Accountant):
    """Privacy loss of one background participant who is a unit in K released models, each run for `steps` steps:
    RDP composes additively over the K*steps Poisson-subsampled Gaussian steps."""
    def __init__(self, q, sigma, delta, K):
        super().__init__(q, sigma, delta); self.K = K
    def eps(self, steps, method='rdp'):
        return super().eps(steps * self.K, method)

def sigma_for_population(K, target_eps_init, q=0.2, delta=1e-5, steps=150):
    """Smallest noise multiplier such that K models' enrolment training composes to <= target_eps_init."""
    lo, hi = 1.0, 200.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if Accountant(q, mid, delta).eps(K * steps) > target_eps_init: lo = mid
        else: hi = mid
    return hi

# ------------------------------------------------------------------ motor noise (CMU feature layout H, DD, UD x10 + H.Return)
def motor_noise(x, rng, cv=0.08, floor=0.005):
    x = x.copy(); d = len(x)
    if d == 31:
        H = np.r_[np.arange(0, 30, 3), 30]; UD = np.arange(2, 30, 3); DD = np.arange(1, 30, 3)
        for idx in (H, UD): x[idx] = x[idx] + rng.normal(0, cv * np.abs(x[idx]) + floor)
        x[H] = np.maximum(x[H], 0.001)
        x[DD] = x[DD - 1] + x[DD + 1]                      # keep DD = H(prev key) + UD consistent
    else:
        x = x + rng.normal(0, cv * np.abs(x) + floor)
    return x

# ------------------------------------------------------------------ prequential simulation (extended)
def run_stream2(run, cfg, method, acct, cond, seed, collect_trace=False, scores=False):
    kind, rate = CONDITIONS2[cond]
    if cond == 'C3s_oracle_strong_40': cfg = dict(cfg, burst=2 * cfg['burst'])
    rng = np.random.default_rng(seed)
    out_rng = np.random.default_rng(seed + 17)
    aux = np.random.default_rng(seed * 7 + 12345)              # separate stream: audit, handover, noise, generative
    t0 = time.perf_counter()
    A = Authenticator2(run['anchor'], run['bg_units'], run['cal_g'], run['cal_bg'], cfg, method, acct, rng)
    A.fit(cfg['init_steps']); t_init = time.perf_counter() - t0
    psr_init = float(np.mean(A.score(run['att_hold']) >= A.tau_acc))
    Xs, ss = run['stream_X'], run['stream_s']; n = len(Xs); horizon = n
    plan = set(attack_plan(n, rate, cfg['burst'], seed + 7919 * sid(run['victim']))) if kind != 'handover' else set()
    apool = run['att_pool_raw']; aptr = 0; alpha = 0.0; vleak = run['leak_raw']
    if kind == 'generative':                                    # diagonal Gaussian victim model in transformed space
        Lt = run['transform'](vleak); g_mu, g_sd = Lt.mean(0), Lt.std(0) + 0.05 * np.abs(Lt.mean(0)) + 1e-3
        def inv_t(Y): return np.sign(Y) * 0.01 * np.expm1(np.abs(Y)) if run['transform'] is cmu_transform else np.sign(Y) * np.expm1(np.abs(Y))
    mmin = cfg['m_min'] if method['silent'] == 'risk' or method['pacing'] else 1
    G_win = cfg.get('grace_window', 10) if method.get('grace') else 0
    rec = []; ep = []; upd_t = []; ckpt = []; inf_t = []; tcount = [0]; grace_left = [0]; pending_handover = [False]
    def flush():
        if method['adaptive'] and ep:
            X = np.vstack([e[0] for e in ep]); p = np.array([e[1] for e in ep])
            ver = np.array([e[2] for e in ep]); f = np.array([e[3] for e in ep]); idx = [e[4] for e in ep]
            ok = A.admit(X, p, ver)
            if ok.any():
                A.queue(X[ok], f[ok])
                for i in np.where(ok)[0]: rec[idx[i]]['admitted'] = True
            t = time.perf_counter()
            if A.maybe_update(tcount[0], horizon, mmin): upd_t.append(time.perf_counter() - t)
        ep.clear()
    def present(x, is_att, sess):
        t = time.perf_counter(); p = float(A.score(x[None])[0]); inf_t.append(time.perf_counter() - t)
        in_grace = G_win and grace_left[0] > 0
        audit = (not in_grace) and method.get('audit', 0.0) > 0 and p >= A.tau_acc and aux.random() < method['audit']
        if in_grace: dec, ver = 'accept', False                 # trust window: no re-challenge, label inherited
        elif p >= A.tau_acc and not audit: dec, ver = 'accept', False
        else:
            passed = out_rng.random() < (cfg['p_att_pass'] if is_att else cfg['p_gen_pass'])
            dec, ver = ('stepup_pass' if passed else 'stepup_fail'), passed
            if audit: dec = 'audit_' + dec
        lab = ver
        if G_win:
            if in_grace: lab = True; grace_left[0] -= 1
            elif ver: grace_left[0] = G_win
        rec.append(dict(p=p, dec=dec, att=is_att, sess=int(sess), admitted=False, verified=ver, labelled=lab,
                        nov=float(A.novelty(x[None])[0])))
        if lab: A.add_verified(x[None])
        ep.append((x[None], p, lab, is_att, len(rec) - 1)); tcount[0] += 1
        if len(ep) >= cfg['W']: flush()
        return dec
    def burst(sess):
        nonlocal aptr, alpha
        outs = []
        for k in range(cfg['burst']):
            if kind in ('zero', 'handover'): xa = apool[aptr % len(apool)]
            elif kind in ('oracle', 'oracle_noisy', 'gate_aware', 'generative'):
                base = vleak[k % len(vleak)]
                if kind == 'generative': base = inv_t(aux.normal(g_mu, g_sd))
                cand = (1 - ALPHA_GRID[:, None]) * base + ALPHA_GRID[:, None] * apool[aptr % len(apool)]
                Zc = run['Z'](run['transform'](cand)); pc = A.score(Zc)
                okc = pc >= A.tau_upd
                if kind == 'gate_aware' and np.isfinite(A.R): okc &= A.novelty(Zc) <= A.R
                okc = np.where(okc)[0]; xa = cand[okc.max()] if len(okc) else cand[0]
                if kind == 'oracle_noisy': xa = motor_noise(xa, aux)
            else: xa = (1 - alpha) * vleak[k % len(vleak)] + alpha * apool[aptr % len(apool)]
            aptr += 1
            o = present(run['Z'](run['transform'](xa[None]))[0], True, sess); outs.append(o)
            if o.endswith('stepup_fail'): break
        if kind == 'adaptive':
            acc = np.mean([o == 'accept' for o in outs]) if len(outs) == cfg['burst'] else 0.0
            alpha = min(1.0, alpha + cfg['adapt_step_up']) if acc >= 0.8 else max(0.0, alpha - cfg['adapt_step_down'])
    prev_s = None; ho_done = 0; probe = run['probe']
    for i in range(n):
        if ss[i] != prev_s:
            g = A.score(Xs[ss == ss[i]]); im = A.score(probe)
            row = dict(sess=int(ss[i]), eer=eer(g, im), frr=float(np.mean(g < A.tau_acc)),
                       far=float(np.mean(im >= A.tau_acc)), eps=A.eps() if method['dp'] else np.nan,
                       n_updates=A.n_updates, tau_acc=A.tau_acc)
            ims = np.sort(im)[::-1]
            for ft in (0.01, 0.05, 0.10):                      # oracle matched-FAR threshold on probes
                k_ = max(int(np.floor(ft * len(ims))), 1); thr = ims[k_ - 1] + 1e-12
                row[f'frr_at_far{int(ft*100):02d}'] = float(np.mean(g < thr))
            row['det'] = np.array([np.mean(g < (ims[max(int(np.floor(f * len(ims))), 1) - 1] + 1e-12)) for f in FAR_GRID])
            ckpt.append(row); prev_s = ss[i]
            if G_win: grace_left[0] = G_win                  # session sign-in with the strong factor opens a grace window
            if kind == 'handover': burst(ss[i])               # owner signs in, then hands the device over
        if i in plan: burst(ss[i])
        present(Xs[i], False, ss[i])
    flush()
    R = pd.DataFrame(rec); G = R[~R.att]; Aa = R[R.att]
    late = G.sess >= (G.sess.max() - 2)
    delays = []
    if len(Aa):
        idx = np.where(R.att.values)[0]; starts = [idx[0]] + [j for a_, j in zip(idx[:-1], idx[1:]) if j != a_ + 1]
        for st in starts:
            d = 0
            while st + d < len(R) and R.att.values[st + d] and R.dec.values[st + d] == 'accept': d += 1
            delays.append(d)
    ck = pd.DataFrame(ckpt); det_late = np.vstack(ck.det.iloc[-3:].values).mean(0); ck = ck.drop(columns='det')
    Gad = G[G.admitted]
    out = dict(
        online_FRR=float(np.mean(G.dec != 'accept')), online_FRR_late=float(np.mean(G[late].dec != 'accept')),
        genuine_stepup_rate=float(np.mean(G.dec != 'accept')),
        genuine_lockout_rate=float(np.mean(G.dec.str.endswith('stepup_fail'))),
        attack_accept_rate=float(np.mean(Aa.dec == 'accept')) if len(Aa) else np.nan,
        attack_samples_accepted=int((Aa.dec == 'accept').sum()) if len(Aa) else 0,
        attack_stepup_passed=int(Aa.dec.str.endswith('stepup_pass').sum()) if len(Aa) else 0,
        detection_delay=float(np.mean(delays)) if delays else np.nan,
        n_updates=A.n_updates, n_admitted=int(R.admitted.sum()),
        n_admitted_attack=int(Aa.admitted.sum()) if len(Aa) else 0,
        n_admitted_verified=int(G[G.admitted].verified.sum()) if len(G) else 0,
        buffer_contam_final=float(A.bufflag.mean()) if len(A.bufflag) else 0.0,
        PSR_init=psr_init, PSR_final=float(np.mean(A.score(run['att_hold']) >= A.tau_acc)),
        eps_final=A.eps() if method['dp'] else np.nan,
        eps_final_pld=acct.eps(A.steps, 'pld') if method['dp'] else np.nan,
        max_unit_grad_norm=float(np.max(A.clip_stats)) if A.clip_stats else np.nan,
        ckpt_EER_mean=float(ck.eer.iloc[1:].mean()), ckpt_EER_late=float(ck.eer.iloc[-3:].mean()),
        ckpt_FAR_late=float(ck.far.iloc[-3:].mean()), ckpt_FRR_late=float(ck.frr.iloc[-3:].mean()),
        FRR_at_FAR01_late=float(ck.frr_at_far01.iloc[-3:].mean()), FRR_at_FAR05_late=float(ck.frr_at_far05.iloc[-3:].mean()),
        FRR_at_FAR10_late=float(ck.frr_at_far10.iloc[-3:].mean()), tau_acc_final=A.tau_acc, tau_upd_final=A.tau_upd,
        p_admitted_genuine=float(Gad.p.mean()) if len(Gad) else np.nan, nov_admitted_genuine=float(Gad.nov.mean()) if len(Gad) else np.nan,
        p_genuine_all=float(G.p.mean()), nov_genuine_all=float(G.nov.mean()),
        t_init_s=t_init, t_update_s=float(np.mean(upd_t)) if upd_t else np.nan,
        t_infer_us=float(np.median(inf_t) * 1e6), steps=A.steps, n_stream=len(R), n_attack=int(len(Aa)),
        upd_per_session=','.join(str(int(v)) for v in np.diff(np.r_[ck.n_updates.values, A.n_updates])),
    )
    for j, v in enumerate(det_late): out[f'det{j:02d}'] = float(v)
    if collect_trace: return out, R, ck, A
    return out, ck

# ------------------------------------------------------------------ data variants
def rff_wrap(run, D=256, ls=0.5, seed=2024):
    """Non-linear verifier: fixed, data-independent random Fourier features approximating an RBF kernel with
    lengthscale ls*sqrt(d) on the standardised features. D and ls were chosen on the pilot partition (seed 999) by
    static-model EER only (D=256, ls=0.5: 0.135 vs 0.145 for the linear verifier). Being data-independent, the map
    leaves the DP guarantee unchanged; the model then has D+1 weights."""
    d = run['anchor'].shape[1]; r = np.random.default_rng(seed)
    W = r.normal(0, 1 / (ls * np.sqrt(d)), size=(d, D)); b = r.uniform(0, 2 * np.pi, D)
    phi = lambda Z: np.sqrt(2.0 / D) * np.cos(Z @ W + b)
    out = dict(run); Z0 = run['Z']
    out['Z'] = lambda X: phi(Z0(X))
    for k in ('anchor', 'cal_g', 'stream_X', 'cal_bg', 'att_hold', 'probe'): out[k] = phi(run[k])
    out['bg_units'] = [phi(U) for U in run['bg_units']]
    return out

def inv_cmu(Y): return np.sign(Y) * 0.01 * np.expm1(np.abs(Y))

def drift_variant(data, mode, seed=0, transform=cmu_transform, inv=inv_cmu):
    """mode='detrended': remove each subject's per-session mean shift (practice effect) in transformed space.
    mode='randomwalk': detrended + a per-subject, per-feature Gaussian random walk across sessions whose
    session-8 displacement is matched to the original data's median displacement (non-monotone 'ageing')."""
    r = np.random.default_rng(seed); out = {}
    disp_orig = []
    for s, v in data.items():
        Y = transform(v['X_raw']); ss = v['sess']; m1, sd1 = Y[ss == 1].mean(0), Y[ss == 1].std(0) + 1e-6
        disp_orig.append(np.sqrt(np.mean(((Y[ss == ss.max()].mean(0) - m1) / sd1) ** 2)))
    target = float(np.median(disp_orig))
    for s, v in data.items():
        Y = transform(v['X_raw']).copy(); ss = v['sess']; m1 = Y[ss == 1].mean(0); sd1 = Y[ss == 1].std(0) + 1e-6
        for k in np.unique(ss): Y[ss == k] += m1 - Y[ss == k].mean(0)
        if mode == 'randomwalk':
            K = ss.max(); steps = r.normal(0, 1, size=(K - 1, Y.shape[1]))
            walk = np.vstack([np.zeros(Y.shape[1]), np.cumsum(steps, 0)])
            scale = target / (np.sqrt(np.mean(walk[-1] ** 2)) + 1e-9)
            for k in range(2, K + 1): Y[ss == k] += walk[k - 1] * scale * sd1
        out[s] = dict(X_raw=inv(Y), sess=ss)
    return out, target

# ------------------------------------------------------------------ parallel grid
_G = {}
def _job(args):
    sd, v, att, part = args['sd'], args['v'], args['att'], args['part']
    data, transform, cfg, methods, conds, tag = _G['data'], _G['transform'], args['cfg'], args['methods'], args['conds'], args['tag']
    key = (id(_G['acct_factory']), cfg['q'], cfg['sigma'], cfg['delta'])
    cache = _G.setdefault('acct_cache', {})
    if key not in cache: cache[key] = _G['acct_factory'](cfg)        # per-process accountant cache (RDP/PLD are costly)
    acct = cache[key]
    run = build_run(data, transform, part, v, att, cfg, seed=sd)
    if _G.get('wrap'): run = _G['wrap'](run)
    rows, cks = [], []
    for cond in conds:
        for mn in methods:
            out, ck = run_stream2(run, cfg, METHODS2[mn], acct, cond, seed=sd)
            rows.append(dict(tag=tag, seed=sd, victim=str(v), attacker=str(att), cond=cond, method=mn, **out))
            cks.append(ck.assign(tag=tag, seed=sd, victim=str(v), cond=cond, method=mn))
    return rows, cks

def default_acct(cfg): return Accountant(cfg['q'], cfg['sigma'], cfg['delta'])

def run_grid2(data, transform, seeds, methods, conds, cfg, part_sizes, tag, victims_fixed=None, n_jobs=2,
              acct_factory=default_acct, wrap=None, max_victims=None):
    jobs = []
    for sd in seeds:
        if victims_fixed is None: part = partition(list(data), *part_sizes, seed=sd)
        else:
            others = [s for s in data if s not in victims_fixed]; part = partition(others, *part_sizes, seed=sd); part['victims'] = list(victims_fixed)
        for v in part['victims'][:max_victims]:
            jobs.append(dict(sd=sd, v=v, att=part['attackers'][sid(v) % len(part['attackers'])], part=part, cfg=cfg,
                             methods=methods, conds=conds, tag=tag))
    _G.update(data=data, transform=transform, acct_factory=acct_factory, wrap=wrap, acct_cache={})
    t0 = time.time()
    if n_jobs > 1:
        with mp.get_context('fork').Pool(n_jobs) as pool: res = pool.map(_job, jobs, chunksize=1)
    else: res = [_job(j) for j in jobs]
    rows = [r for rr, _ in res for r in rr]; cks = [c for _, cc in res for c in cc]
    print(f'[{tag}] {len(rows)} runs in {time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(rows), pd.concat(cks, ignore_index=True)

# ------------------------------------------------------------------ statistics
def tost_paired(d, margin, alpha=0.05):
    """Two one-sided Wilcoxon signed-rank tests of |mean paired difference| < margin (equivalence)."""
    from scipy.stats import wilcoxon
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    p_lo = wilcoxon(d + margin, alternative='greater').pvalue      # H0: d <= -margin
    p_hi = wilcoxon(d - margin, alternative='less').pvalue         # H0: d >= +margin
    from scipy.stats import ttest_1samp
    t_lo = ttest_1samp(d, -margin, alternative='greater').pvalue; t_hi = ttest_1samp(d, margin, alternative='less').pvalue
    return dict(margin=margin, p_tost_wilcoxon=max(p_lo, p_hi), p_tost_t=max(t_lo, t_hi),
                ci90_lo=float(np.mean(d) - 1.645 * d.std(ddof=1) / np.sqrt(len(d))),
                ci90_hi=float(np.mean(d) + 1.645 * d.std(ddof=1) / np.sqrt(len(d))),
                equivalent=max(p_lo, p_hi) < alpha)

def mde(sd, n, alpha=0.05, power=0.8, are=3 / np.pi):
    from scipy.stats import norm
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd / np.sqrt(n * are))
