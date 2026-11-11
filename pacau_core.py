# ==== pacau_core: Privacy-budget-Aware, risk-gated Continuous AUthentication ====
import numpy as np, pandas as pd, time, logging, warnings
logging.disable(logging.WARNING)
from dp_accounting import dp_event, rdp, pld

# ---------------------------------------------------------------- data
CMU_FEATURES = None

def load_cmu(path):
    df = pd.read_csv(path)
    assert df.shape == (20400, 34), df.shape
    assert df.isna().sum().sum() == 0
    df = df.sort_values(['subject', 'sessionIndex', 'rep']).reset_index(drop=True)
    feats = [c for c in df.columns if c not in ('subject', 'sessionIndex', 'rep')]
    data = {}
    for s, d in df.groupby('subject', sort=True):
        data[s] = dict(X_raw=d[feats].to_numpy(float), sess=d['sessionIndex'].to_numpy(int))
    return data, feats, df

def cmu_transform(X):
    # a-priori, data-independent transform: signed log of timing in 10 ms units
    return np.sign(X) * np.log1p(np.abs(X) / 0.01)

TOUCH_COLS = ['user', 'doc', 'inter_stroke_time', 'stroke_duration', 'start_x', 'start_y', 'stop_x', 'stop_y',
              'direct_e2e_dist', 'mean_resultant_length', 'direction_flag', 'e2e_direction', 'phone_id',
              'vel_p20', 'vel_p50', 'vel_p80', 'acc_p20', 'acc_p50', 'acc_p80', 'med_vel_last3',
              'max_dev_e2e', 'dev_p20', 'dev_p50', 'dev_p80', 'avg_direction', 'traj_length', 'ratio_e2e_traj',
              'avg_velocity', 'med_acc_first5', 'mid_pressure', 'mid_area', 'mid_finger_orient',
              'finger_orient_change', 'phone_orientation']

def touch_transform(X):
    return np.sign(X) * np.log1p(np.abs(X))

def load_touch(feat_path, raw_path, window=5):
    f = pd.read_csv(feat_path, header=None, low_memory=False)
    f = f.apply(pd.to_numeric, errors='coerce')          # non-numeric cells (header, corrupted rows) -> NaN
    f.columns = TOUCH_COLS
    f = f.replace([np.inf, -np.inf], np.nan)
    n_nan = int(f.isna().any(axis=1).sum()); f = f.dropna()
    r = pd.read_csv(raw_path, header=None)
    r.columns = ['phone', 'user', 'doc', 't', 'action', 'orient', 'x', 'y', 'pressure', 'area', 'forient']
    start = r.groupby(['user', 'doc']).t.min().reset_index()
    start['user'] = start['user'] + 1                    # verified: featMat ids = raw ids + 1 (doc-presence patterns match)
    feats = [c for c in TOUCH_COLS if c not in ('user', 'doc', 'phone_id')]
    data, meta = {}, []
    for u, d in f.groupby('user', sort=True):
        u = int(u)
        st = start[start.user == u].copy(); st['doc_feat'] = st['doc'] + 1   # raw doc 0..6 == featMat doc 1..7
        order = st.sort_values('t')
        docs_in_order = [int(x) for x in order.doc_feat if x in set(d.doc.astype(int))]
        t0 = order.t.min()
        Xs, sess = [], []
        for k, doc in enumerate(docs_in_order):
            dd = d[d.doc == doc][feats].to_numpy(float)          # stroke order within doc preserved
            nw = len(dd) // window
            if nw == 0: continue
            W = dd[:nw * window].reshape(nw, window, -1).mean(1)
            Xs.append(W); sess += [k + 1] * nw
            tdoc = float(order[order.doc_feat == doc].t.iloc[0])
            meta.append(dict(user=u, doc=doc, order=k + 1, n_windows=nw, days_from_first=(tdoc - t0) / 864e5))
        data[u] = dict(X_raw=np.vstack(Xs), sess=np.array(sess))
    return data, feats, pd.DataFrame(meta), n_nan

# ---------------------------------------------------------------- privacy accounting
class Accountant:
    """Wraps Google dp_accounting. Cumulative composition of Poisson-subsampled Gaussian steps."""
    def __init__(self, q, sigma, delta):
        self.q, self.sigma, self.delta = q, sigma, delta
        self._cache = {}
    def eps(self, steps, method='rdp'):
        if steps == 0: return 0.0
        key = (steps, method)
        if key not in self._cache:
            ev = dp_event.PoissonSampledDpEvent(self.q, dp_event.GaussianDpEvent(self.sigma))
            a = rdp.RdpAccountant() if method == 'rdp' else pld.PLDAccountant()
            a.compose(ev, int(steps)); self._cache[key] = float(a.get_epsilon(self.delta))
        return self._cache[key]

# ---------------------------------------------------------------- model + (DP-)SGD
def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def add_bias(X): return np.hstack([X, np.ones((len(X), 1))])

def sgd_train(w, Xb, y, unit, n_units, steps, lr, rng, q, clip=None, sigma=0.0, l2=1e-3, stats=None):
    """Unit-level (DP-)SGD for logistic regression.
    Each privacy unit u contributes g_u = mean_{i in u} grad_i ; g_u is clipped to norm <= clip ;
    units are Poisson-sampled with rate q ; Gaussian noise N(0, (sigma*clip)^2 I) is added to the sum ;
    the noisy sum is divided by the expected batch size q*n_units.  clip=None, sigma=0 -> non-private
    reference with the identical optimiser."""
    d = Xb.shape[1]
    cnt = np.bincount(unit, minlength=n_units).astype(float)
    exp_b = max(q * n_units, 1.0)
    for _ in range(steps):
        sel_u = rng.random(n_units) < q
        m = sel_u[unit]
        if m.any():
            Xs, ys, us = Xb[m], y[m], unit[m]
            g_i = (sigmoid(Xs @ w) - ys)[:, None] * Xs
            G = np.zeros((n_units, d))
            np.add.at(G, us, g_i)
            G = G[sel_u] / cnt[sel_u][:, None]
            if clip is not None:
                nrm = np.linalg.norm(G, axis=1)
                if stats is not None: stats.append(float(nrm.max()))
                G = G * np.minimum(1.0, clip / np.maximum(nrm, 1e-12))[:, None]
            g = G.sum(0)
        else:
            g = np.zeros(d)
        if sigma > 0: g = g + rng.normal(0.0, sigma * clip, size=d)
        w = w - lr * (g / exp_b + l2 * w)
    return w

def platt_fit(z, y, iters=100, ridge=1e-2):
    """Class-balanced Platt scaling with Platt's (1999) smoothed targets and a ridge penalty on the
    slope, solved by damped Newton steps; avoids divergence on (near-)separable calibration sets."""
    npos, nneg = max(y.sum(), 1), max((1 - y).sum(), 1)
    t = np.where(y == 1, (npos + 1) / (npos + 2), 1 / (nneg + 2))
    wts = np.where(y == 1, 0.5 / npos, 0.5 / nneg)
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = sigmoid(a * z + b); r = wts * (p - t); s_ = wts * p * (1 - p) + 1e-12
        g = np.array([np.sum(r * z) + ridge * (a - 1.0), np.sum(r)])
        H = np.array([[np.sum(s_ * z * z) + ridge, np.sum(s_ * z)], [np.sum(s_ * z), np.sum(s_) + 1e-9]])
        step = np.linalg.solve(H, g); step = step * min(1.0, 2.0 / (np.abs(step).max() + 1e-12))
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-7: break
    return float(a), float(b)

def eer(gen, imp):
    """Threshold-free EER (tie-aware: thresholds placed only at distinct score values)."""
    thr = np.unique(np.concatenate([gen, imp]))
    frr = np.searchsorted(np.sort(gen), thr, side='left') / len(gen)          # gen < thr rejected
    far = 1 - np.searchsorted(np.sort(imp), thr, side='left') / len(imp)      # imp >= thr accepted
    i = np.argmin(np.abs(frr - far)); return float((frr[i] + far[i]) / 2)

def auc(pos, neg):
    from scipy.stats import mannwhitneyu
    if len(pos) == 0 or len(neg) == 0: return np.nan
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))

# ---------------------------------------------------------------- authenticator
DEFAULT_CFG = dict(
    n_enroll=40, n_cal=10, n_bg=50, q=0.2, sigma=3.0, clip=1.0, delta=1e-5, lr=0.05, l2=1e-3,
    init_steps=150, upd_steps=25, buffer_cap=100, W=10, tau_acc=0.5, tau_upd=0.8,
    rho=0.8, m_min=5, trust_q=0.95, eps_budget=8.0, burst=10, attack_holdout=0.3,
    p_gen_pass=1.0, p_att_pass=0.0, recal_verified=20,                       # strong-factor (step-up) outcome model (assumption A1)
    adapt_step_up=0.15, adapt_step_down=0.10, n_leak=5, probe_per_subject=20,
)

# silent: how silently-accepted (unlabelled) samples may enter the model
#   'all' = every accepted sample; 'double' = p>=tau_upd; 'risk' = proposed gate; 'none' = never
# labels: whether strong-factor-verified samples (passed step-up) are admissible
METHODS = {
  'B0_static':         dict(adaptive=False, dp=False, silent='none',   labels=False, budget=False, pacing=False, novelty=False, evidence=False, calib=True),
  'B0_static_DP':      dict(adaptive=False, dp=True,  silent='none',   labels=False, budget=False, pacing=False, novelty=False, evidence=False, calib=True),
  'B1_selfupdate':     dict(adaptive=True,  dp=False, silent='all',    labels=True,  budget=False, pacing=False, novelty=False, evidence=False, calib=True),
  'B1b_double_thr':    dict(adaptive=True,  dp=False, silent='double', labels=True,  budget=False, pacing=False, novelty=False, evidence=False, calib=True),
  'B1c_labels_only':   dict(adaptive=True,  dp=False, silent='none',   labels=True,  budget=False, pacing=False, novelty=False, evidence=False, calib=True),
  'B2_DP_selfupdate':  dict(adaptive=True,  dp=True,  silent='all',    labels=True,  budget=True,  pacing=False, novelty=False, evidence=False, calib=True),
  'B2c_DP_labels_only':dict(adaptive=True,  dp=True,  silent='none',   labels=True,  budget=True,  pacing=True,  novelty=False, evidence=False, calib=True),
  'B3_gate_nonDP':     dict(adaptive=True,  dp=False, silent='risk',   labels=True,  budget=False, pacing=False, novelty=True,  evidence=True,  calib=True),
  'P_gate_DP_budget':  dict(adaptive=True,  dp=True,  silent='risk',   labels=True,  budget=True,  pacing=True,  novelty=True,  evidence=True,  calib=True),
  'A_P_no_budget':     dict(adaptive=True,  dp=True,  silent='risk',   labels=True,  budget=False, pacing=False, novelty=True,  evidence=True,  calib=True),
  'A_P_no_pacing':     dict(adaptive=True,  dp=True,  silent='risk',   labels=True,  budget=True,  pacing=False, novelty=True,  evidence=True,  calib=True),
  'A_P_no_novelty':    dict(adaptive=True,  dp=True,  silent='risk',   labels=True,  budget=True,  pacing=True,  novelty=False, evidence=True,  calib=True),
  'A_P_no_evidence':   dict(adaptive=True,  dp=True,  silent='risk',   labels=True,  budget=True,  pacing=True,  novelty=True,  evidence=False, calib=True),
  'A_P_no_calib':      dict(adaptive=True,  dp=True,  silent='risk',   labels=True,  budget=True,  pacing=True,  novelty=True,  evidence=True,  calib=False),
}

class Authenticator:
    def __init__(self, Xg_anchor, Xbg_units, Xcal_g, Xcal_bg, cfg, method, acct, rng):
        self.c, self.m, self.acct, self.rng = cfg, method, acct, rng
        self.anchor = Xg_anchor; self.buf = np.zeros((0, Xg_anchor.shape[1])); self.bufflag = np.zeros(0, bool)
        self.bg = Xbg_units                      # list of arrays, one per background participant
        self.Xcal = np.vstack([Xcal_g] + [Xcal_bg]); self.ycal = np.r_[np.ones(len(Xcal_g)), np.zeros(len(Xcal_bg))]
        self.steps = 0; self.n_updates = 0; self.w = np.zeros(Xg_anchor.shape[1] + 1)
        # anchored trust region (diagonal Mahalanobis on enrolment data, computed on-device)
        G = np.vstack([Xg_anchor, Xcal_g]); self.nmu = Xg_anchor.mean(0); self.nsd = Xg_anchor.std(0) + 1e-6
        self.R = np.quantile(self.novelty(G), cfg['trust_q']) if method['novelty'] else np.inf
        self.clip_stats = [] if method['dp'] else None
        self.pend = np.zeros((0, Xg_anchor.shape[1])); self.pendflag = np.zeros(0, bool)
        self.last_update_t = 0; self.verified = np.zeros((0, Xg_anchor.shape[1]))
    def novelty(self, X): return np.sqrt(np.mean(((X - self.nmu) / self.nsd) ** 2, axis=1))
    def _dataset(self):
        Xs = list(self.bg) + [np.vstack([self.anchor, self.buf])]
        nb = len(self.bg); G = Xs[-1]
        X = np.vstack(Xs); y = np.r_[np.zeros(sum(len(b) for b in self.bg)), np.ones(len(G))]
        unit = np.r_[np.concatenate([np.full(len(b), i) for i, b in enumerate(self.bg)]), nb + np.arange(len(G))].astype(int)
        return add_bias(X), y, unit, nb + len(G)
    def fit(self, steps):
        Xb, y, unit, nu = self._dataset(); c = self.c
        if self.m['dp']:
            self.w = sgd_train(self.w, Xb, y, unit, nu, steps, c['lr'], self.rng, c['q'], clip=c['clip'], sigma=c['sigma'], l2=c['l2'], stats=self.clip_stats)
        else:
            self.w = sgd_train(self.w, Xb, y, unit, nu, steps, c['lr'], self.rng, c['q'], clip=None, sigma=0.0, l2=c['l2'])
        self.steps += steps
        Xc, yc = self.Xcal, self.ycal
        if self.m['labels'] and len(self.verified) and self.c['recal_verified'] > 0:
            V = self.verified[-self.c['recal_verified']:]
            Xc, yc = np.vstack([Xc, V]), np.r_[yc, np.ones(len(V))]
        self.a, self.b = platt_fit(add_bias(Xc) @ self.w, yc) if self.m['calib'] else (1.0, 0.0)
    def eps(self, steps=None):
        return self.acct.eps(self.steps if steps is None else steps) if self.m['dp'] else np.inf
    def n_affordable(self):
        k = 0
        while self.acct.eps(self.steps + (k + 1) * self.c['upd_steps']) <= self.c['eps_budget'] and k < 10000: k += 1
        return k
    def can_spend(self, t, horizon):
        if not (self.m['dp'] and self.m['budget']): return True
        if self.acct.eps(self.steps + self.c['upd_steps']) > self.c['eps_budget']: return False
        if self.m['pacing']:                                   # spread the remaining budget over the horizon
            if not hasattr(self, 'min_gap'): self.min_gap = horizon / max(self.n_affordable(), 1)
            return (t - self.last_update_t) >= self.min_gap
        return True
    def score(self, X): return sigmoid(self.a * (add_bias(X) @ self.w) + self.b)
    def admit(self, X, p, verified):
        """Update eligibility for one episode -- a decision separate from access.
        verified[i] = True if sample i was challenged and passed the strong factor (labelled genuine)."""
        c, m = self.c, self.m
        lab = verified & m['labels']
        sil = (p >= c['tau_acc']) & ~verified                      # silently accepted, unlabelled
        if m['silent'] == 'none': ok_s = np.zeros(len(p), bool)
        elif m['silent'] == 'all': ok_s = sil
        elif m['silent'] == 'double': ok_s = sil & (p >= c['tau_upd'])
        else:                                                      # proposed risk gate
            ok_s = sil & (p >= c['tau_upd']) & (self.novelty(X) <= self.R)
            if m['evidence'] and not ((np.mean(p >= c['tau_acc']) >= c['rho']) and (np.median(p) >= c['tau_upd'])):
                ok_s[:] = False
        return lab | ok_s
    def add_verified(self, x): self.verified = np.vstack([self.verified, x])[-200:]
    def queue(self, Xnew, flags):
        self.pend = np.vstack([self.pend, Xnew]); self.pendflag = np.r_[self.pendflag, flags]
    def maybe_update(self, t, horizon, mmin):
        if len(self.pend) < mmin or not self.can_spend(t, horizon): return False
        self.buf = np.vstack([self.buf, self.pend])[-self.c['buffer_cap']:]
        self.bufflag = np.r_[self.bufflag, self.pendflag][-self.c['buffer_cap']:]
        self.pend = self.pend[:0]; self.pendflag = self.pendflag[:0]
        self.fit(self.c['upd_steps']); self.n_updates += 1; self.last_update_t = t
        return True

# ---------------------------------------------------------------- scenario construction (leakage-safe)
def sid(s):
    """Stable integer id for subject labels such as 's046' or 41."""
    d = ''.join(ch for ch in str(s) if ch.isdigit()); return int(d) if d else abs(hash(str(s))) % 10**6

def partition(subjects, n_bgtrain, n_bgcal, n_att, seed):
    r = np.random.default_rng(seed); s = np.array(sorted(subjects)); r.shuffle(s)
    a, b, c = n_bgtrain, n_bgtrain + n_bgcal, n_bgtrain + n_bgcal + n_att
    return dict(bg_train=list(s[:a]), bg_cal=list(s[a:b]), attackers=list(s[b:c]), victims=list(s[c:]))

def build_run(data, transform, part, victim, attacker, cfg, seed):
    """All samples of one (seed, victim) run. Standardiser + Platt reference = bg_cal (public reference set)."""
    r = np.random.default_rng(seed * 100003 + sid(victim))
    T = lambda s: transform(data[s]['X_raw'])
    ref = np.vstack([T(s)[r.choice(len(data[s]['sess']), min(cfg['n_bg'], len(data[s]['sess'])), replace=False)] for s in part['bg_cal']])
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    Z = lambda X: (X - mu) / sd
    Xv, sv = Z(T(victim)), data[victim]['sess']
    ne, nc = cfg['n_enroll'], cfg['n_cal']
    run = dict(victim=victim, attacker=attacker, mu=mu, sd=sd, Z=Z,
               anchor=Xv[:ne], cal_g=Xv[ne:ne + nc], stream_X=Xv[ne + nc:], stream_s=sv[ne + nc:],
               cal_bg=Z(ref))
    run['bg_units'] = [Z(T(s)[r.choice(len(data[s]['sess']), min(cfg['n_bg'], len(data[s]['sess'])), replace=False)]) for s in part['bg_train']]
    Xa_raw = data[attacker]['X_raw']; na = len(Xa_raw); cut = int(round(na * (1 - cfg['attack_holdout'])))
    run['att_pool_raw'] = Xa_raw[:cut]; run['att_hold'] = Z(transform(Xa_raw[cut:]))
    run['leak_raw'] = data[victim]['X_raw'][:ne][r.choice(ne, cfg['n_leak'], replace=False)]
    probes = [p for p in part['attackers'] if p != attacker]
    run['probe'] = np.vstack([Z(T(p))[r.choice(len(data[p]['sess']), cfg['probe_per_subject'], replace=False)] for p in probes])
    run['transform'] = transform
    return run

def attack_plan(n_stream, rate, burst, seed):
    """Burst start positions (indices into the genuine stream before which a burst is inserted)."""
    if rate <= 0: return []
    r = np.random.default_rng(seed); nb = int(round(rate * n_stream / burst))
    lo = min(10, n_stream // 10)
    return sorted(r.choice(np.arange(lo, n_stream), nb, replace=False).tolist())

CONDITIONS = {'C0_drift_only': ('none', 0.0), 'C1_zero_effort_10': ('zero', 0.10),
              'C2_zero_effort_30': ('zero', 0.30), 'C3_confaware_oracle_20': ('oracle', 0.20),
              'C4_greybox_feedback_20': ('adaptive', 0.20)}
ALPHA_GRID = np.linspace(0, 1, 21)

def run_stream(run, cfg, method, acct, cond, seed, collect_trace=False):
    """Prequential (test-then-train) simulation of one victim's chronological stream.
    Access decision: accept (p >= tau_acc) or challenge with a strong factor (step-up).
    A genuine user passes the challenge w.p. p_gen_pass (sample becomes *verified*); an attacker
    passes w.p. p_att_pass (default 0) -- failing locks the session and ends the attack burst.
    Update decision: taken per episode of W presented samples by Authenticator.admit(), and applied
    by maybe_update() subject to m_min, privacy budget and pacing."""
    kind, rate = CONDITIONS[cond]
    rng = np.random.default_rng(seed)
    out_rng = np.random.default_rng(seed + 17)               # step-up outcome randomness
    t0 = time.perf_counter()
    A = Authenticator(run['anchor'], run['bg_units'], run['cal_g'], run['cal_bg'], cfg, method, acct, rng)
    A.fit(cfg['init_steps']); t_init = time.perf_counter() - t0
    psr_init = float(np.mean(A.score(run['att_hold']) >= cfg['tau_acc']))
    Xs, ss = run['stream_X'], run['stream_s']; n = len(Xs); horizon = n
    plan = set(attack_plan(n, rate, cfg['burst'], seed + 7919 * sid(run['victim'])))
    apool = run['att_pool_raw']; aptr = 0; alpha = 0.0; vleak = run['leak_raw']
    mmin = cfg['m_min'] if method['silent'] == 'risk' or method['pacing'] else 1
    rec = []; ep = []; upd_t = []; ckpt = []; inf_t = []; tcount = [0]
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
        if p >= cfg['tau_acc']: dec, ver = 'accept', False
        else:
            passed = out_rng.random() < (cfg['p_att_pass'] if is_att else cfg['p_gen_pass'])
            dec, ver = ('stepup_pass' if passed else 'stepup_fail'), passed
        rec.append(dict(p=p, dec=dec, att=is_att, sess=int(sess), admitted=False, verified=ver))
        if ver: A.add_verified(x[None])
        ep.append((x[None], p, ver, is_att, len(rec) - 1)); tcount[0] += 1
        if len(ep) >= cfg['W']: flush()
        return dec
    prev_s = None
    for i in range(n):
        if ss[i] != prev_s:                                   # checkpoint: model state before a new session
            g = A.score(Xs[ss == ss[i]]); im = A.score(run['probe'])
            ckpt.append(dict(sess=int(ss[i]), eer=eer(g, im), frr=float(np.mean(g < cfg['tau_acc'])),
                             far=float(np.mean(im >= cfg['tau_acc'])), eps=A.eps() if method['dp'] else np.nan,
                             n_updates=A.n_updates))
            prev_s = ss[i]
        if i in plan:                                         # attacker obtains the unlocked device: one burst
            outs = []
            for k in range(cfg['burst']):
                if kind == 'zero': xa = apool[aptr % len(apool)]
                elif kind == 'oracle':   # white-box confidence-aware: largest step toward own behaviour that the
                    cand = (1 - ALPHA_GRID[:, None]) * vleak[k % len(vleak)] + ALPHA_GRID[:, None] * apool[aptr % len(apool)]
                    pc = A.score(run['Z'](run['transform'](cand)))       # model still scores >= tau_upd
                    okc = np.where(pc >= cfg['tau_upd'])[0]; xa = cand[okc.max()] if len(okc) else cand[0]
                else: xa = (1 - alpha) * vleak[k % len(vleak)] + alpha * apool[aptr % len(apool)]
                aptr += 1
                o = present(run['Z'](run['transform'](xa[None]))[0], True, ss[i]); outs.append(o)
                if o == 'stepup_fail': break                  # locked out: burst ends
            if kind == 'adaptive':                            # decision-feedback hill climbing (grey box)
                acc = np.mean([o == 'accept' for o in outs]) if len(outs) == cfg['burst'] else 0.0
                alpha = min(1.0, alpha + cfg['adapt_step_up']) if acc >= 0.8 else max(0.0, alpha - cfg['adapt_step_down'])
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
    ck = pd.DataFrame(ckpt)
    out = dict(
        online_FRR=float(np.mean(G.dec != 'accept')), online_FRR_late=float(np.mean(G[late].dec != 'accept')),
        genuine_stepup_rate=float(np.mean(G.dec != 'accept')),
        attack_accept_rate=float(np.mean(Aa.dec == 'accept')) if len(Aa) else np.nan,
        attack_samples_accepted=int((Aa.dec == 'accept').sum()) if len(Aa) else 0,
        detection_delay=float(np.mean(delays)) if delays else np.nan,
        n_updates=A.n_updates, n_admitted=int(R.admitted.sum()),
        n_admitted_attack=int(Aa.admitted.sum()) if len(Aa) else 0,
        n_admitted_verified=int(G[G.admitted].verified.sum()) if len(G) else 0,
        buffer_contam_final=float(A.bufflag.mean()) if len(A.bufflag) else 0.0,
        PSR_init=psr_init, PSR_final=float(np.mean(A.score(run['att_hold']) >= cfg['tau_acc'])),
        eps_final=A.eps() if method['dp'] else np.nan,
        eps_final_pld=acct.eps(A.steps, 'pld') if method['dp'] else np.nan,
        ckpt_EER_mean=float(ck.eer.iloc[1:].mean()), ckpt_EER_late=float(ck.eer.iloc[-3:].mean()),
        ckpt_FAR_late=float(ck.far.iloc[-3:].mean()), ckpt_FRR_late=float(ck.frr.iloc[-3:].mean()),
        t_init_s=t_init, t_update_s=float(np.mean(upd_t)) if upd_t else np.nan,
        t_infer_us=float(np.median(inf_t) * 1e6), steps=A.steps,
        max_unit_grad_norm=float(np.max(A.clip_stats)) if A.clip_stats else np.nan,
        n_stream=len(R), n_attack=int(len(Aa)),
    )
    if collect_trace: return out, R, ck, A
    return out, ck
