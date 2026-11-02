"""Regime and jump indices for any return series (Giorno-Tsallis programme, Papers A and B).

Three questions, one number each, for regimes and for jumps:
  Regimes (Paper A):  kappa      how different are the two regimes          (0..1)
                      T*kappa^2  effective sample size for persistence      (<1 negligible, 1-100 grey, >100 usable)
                      Z, rho     is there persistence?  Z=sqrt(T-1)acf1(d), rho_hat=acf1(d)/kappa
  Jumps (Paper B):    eps, theta incidence and size of one-sided downward impulses (jump-mixture MLE on GARCH-t)
                      T*eps^2*chi2  effective number of jump observations   (same zones)
                      SRJ Z      symmetric-robust jump test (down and up), sup over theta
Both chi2 terms are order-2 Tsallis divergences: kappa=sum_j pi_j D2(f_j||fbar), chi2_J=D2(h^-||f).

Usage: python monitor.py NAME FILE [--freq 5] [--session HH:MM-HH:MM] [--out DIR]
"""
import argparse, ctypes, json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import t as tdist
from scipy.special import roots_genlaguerre, gammaln, logsumexp, expit, logit
from scipy.optimize import minimize, minimize_scalar
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
AP = ctypes.CDLL(os.path.join(HERE, "aparch2.so")); dp = ctypes.POINTER(ctypes.c_double)
AP.aparch_h.restype = ctypes.c_double
AP.aparch_h.argtypes = [dp, ctypes.c_int, dp, ctypes.c_double, dp]
XN, WN = roots_genlaguerre(32, 1); WN = WN / np.exp(gammaln(2.0))   # Z ~ Gamma(2,1)
GRID = np.array([0.3, 0.6, 0.9, 1.5, 2.5])                          # impulse scale, in units of conditional sd

# ---------------------------------------------------------------- data
def load(path):
    head = open(path).readline()
    if head.startswith("<DATE>"):
        d = pd.read_csv(path, sep="\t"); d.columns = [c.strip("<>").lower() for c in d.columns]
        ts = pd.to_datetime(d["date"] + " " + d["time"], format="%Y.%m.%d %H:%M:%S")
    elif head.startswith("DateTime\t"):
        d = pd.read_csv(path, sep="\t"); d.columns = [c.lower() for c in d.columns]
        ts = pd.to_datetime(d["datetime"], format="%Y.%m.%d %H:%M:%S")
    else:
        d = pd.read_csv(path); d.columns = [c.lower() for c in d.columns]
        ts = pd.to_datetime(d["datetime"], format="mixed")
    s = pd.Series(d["close"].astype(float).values, index=ts).sort_index()
    return s[~s.index.duplicated(keep="last")]

def returns(px, freq, session):
    if session:
        a, b = session.split("-"); px = px.between_time(a, b)
    out = []
    for day, g in px.groupby(px.index.date):
        g = g.resample(f"{freq}min").last().ffill()
        r = np.diff(np.log(g.values))
        if len(r) >= 5:
            out.append(pd.Series(r, index=g.index[1:]))
    r = pd.concat(out); return r[np.isfinite(r)]

def deseason(r):
    """divide by time-of-day mean absolute return (intraday U-shape), rescale to unit sd"""
    tod = r.index.hour * 60 + r.index.minute
    s = pd.Series(np.abs(r.values)).groupby(tod).transform("mean").values
    x = r.values / np.where(s > 0, s, np.nan)
    ok = np.isfinite(x); x = x[ok]; idx = r.index[ok]
    return pd.Series((x - x.mean()) / x.std(), index=idx)

# ---------------------------------------------------------------- regime index
def tpdf(y, s, nu): return tdist.pdf(y / s, nu) / s

def fit_mix(y, starts=None):
    """i.i.d. two-component centred Student-t mixture; regime 1 = larger scale"""
    def unpack(q): return expit(q[0]), np.exp(q[1]), np.exp(q[2]), 2.05 + 97.95 * expit(q[3]), 2.05 + 97.95 * expit(q[4])
    def nll(q):
        p0, s0, s1, n0, n1 = unpack(q)
        l = np.logaddexp(np.log(p0) + tdist.logpdf(y / s0, n0) - np.log(s0), np.log1p(-p0) + tdist.logpdf(y / s1, n1) - np.log(s1))
        return -l.mean()
    sd = y.std(); best = None
    st = [[logit(.7), np.log(.6 * sd), np.log(1.6 * sd), logit((8 - 2.05) / 97.95), logit((5 - 2.05) / 97.95)],
          [logit(.5), np.log(.8 * sd), np.log(1.2 * sd), logit((20 - 2.05) / 97.95), logit((4 - 2.05) / 97.95)],
          [logit(.85), np.log(.8 * sd), np.log(2.5 * sd), logit((6 - 2.05) / 97.95), logit((6 - 2.05) / 97.95)]] + (starts or [])
    for q0 in st:
        try: r = minimize(nll, np.array(q0), method="L-BFGS-B", options={"maxiter": 400})
        except Exception: continue
        if best is None or r.fun < best.fun: best = r
    p0, s0, s1, n0, n1 = unpack(best.x)
    if s0 > s1: p0, s0, s1, n0, n1 = 1 - p0, s1, s0, n1, n0
    return dict(pi0=p0, s0=s0, s1=s1, nu0=n0, nu1=n1, ll=-best.fun * len(y), q=list(best.x))

def regime_index(y, m):
    p0, p1 = m["pi0"], 1 - m["pi0"]
    f0 = lambda v: tpdf(v, m["s0"], m["nu0"]); f1 = lambda v: tpdf(v, m["s1"], m["nu1"])
    fb = lambda v: p0 * f0(v) + p1 * f1(v)
    chi = quad(lambda v: (f0(v) - f1(v)) ** 2 / fb(v), -np.inf, np.inf, limit=400)[0]
    kap = p0 * p1 * chi
    d = (f0(y) - f1(y)) / fb(y); d = d - d.mean()
    a1 = float(np.dot(d[:-1], d[1:]) / np.dot(d, d)); T = len(y)
    post1 = p1 * f1(y) / fb(y)                         # P(turbulent | y_t)
    return dict(T=T, kappa=kap, neff=T * kap ** 2, acf1_d=a1, Z=np.sqrt(T - 1) * a1,
                rho_hat=float(np.clip(a1 / kap, -1, 1)) if kap > 0 else np.nan,
                zone="negligible" if T * kap ** 2 < 1 else ("grey" if T * kap ** 2 < 100 else "usable"),
                post_turb_mean=float(post1.mean())), post1

# ---------------------------------------------------------------- GARCH(1,1)-t
def garch_fit(y):
    h0 = float(y.var())
    def full(q): return np.array([q[0], np.exp(q[1]), expit(q[2]) * .999, 0., expit(q[3]) * .999, 2., 2.05 + 97.95 * expit(q[4])])
    def nll(q):
        b = full(q)
        if b[2] + b[4] >= .999: return 1e6
        v = AP.aparch_h(np.ascontiguousarray(y).ctypes.data_as(dp), len(y), np.ascontiguousarray(b).ctypes.data_as(dp), h0, None)
        return -v / len(y) if np.isfinite(v) and v > -1e299 else 1e6
    best = None
    for b0 in ([0., .05 * h0, .06, .92, 6.], [0., .2 * h0, .12, .8, 4.5]):
        q0 = np.array([b0[0], np.log(b0[1]), logit(b0[2] / .999), logit(b0[3] / .999), logit((b0[4] - 2.05) / 97.95)])
        r = minimize(nll, q0, method="L-BFGS-B", options={"maxiter": 500})
        if best is None or r.fun < best.fun: best = r
    b = full(best.x); h = np.empty(len(y))
    AP.aparch_h(np.ascontiguousarray(y).ctypes.data_as(dp), len(y), b.ctypes.data_as(dp), h0, h.ctypes.data_as(dp))
    sc = np.sqrt(h * (b[6] - 2) / b[6])
    return dict(mu=b[0], omega=b[1], alpha=b[2], beta=b[4], nu=b[6], ll=-best.fun * len(y)), sc

# ---------------------------------------------------------------- jump index
def hminus(y, loc, sc, nu, th):          # density of y when a downward impulse th*sd*Z is added
    u = th * sc * np.sqrt(nu / (nu - 2))   # impulse scale in units of the conditional sd
    return sum(w * tpdf(y - loc + u * q, sc, nu) for q, w in zip(XN, WN))
def hplus(y, loc, sc, nu, th):
    u = th * sc * np.sqrt(nu / (nu - 2))
    return sum(w * tpdf(y - loc - u * q, sc, nu) for q, w in zip(XN, WN))

def srj(y, loc, sc, nu, rng, nsim=20000):
    f = tpdf(y - loc, sc, nu); z = (y - loc) / sc
    smu = ((nu + 1) * z / (nu + z * z)) / sc
    A = []
    for th in GRID:
        a = (hminus(y, loc, sc, nu, th) - hplus(y, loc, sc, nu, th)) / f
        A.append(a - np.dot(a, smu) / np.dot(smu, smu) * smu)
    A = np.array(A); Zs = A.sum(1) / np.sqrt((A ** 2).sum(1))
    L = np.linalg.cholesky(np.corrcoef(A) + 1e-10 * np.eye(len(GRID)))
    sims = (L @ rng.standard_normal((len(GRID), nsim))).max(0)
    zd, zu = Zs.max(), (-Zs).max()
    return dict(srj_down=float(zd), p_down=float((sims >= zd).mean()), srj_up=float(zu), p_up=float((sims >= zu).mean()),
                Z_theta=Zs.tolist()), A

def chi2_jump(nu, th):
    """chi^2(h^- || f) in standardised units (unit conditional sd): order-2 Tsallis divergence of the impulse law"""
    s = np.sqrt((nu - 2) / nu)
    f = lambda v: tpdf(v, s, nu)
    h = lambda v: hminus(np.atleast_1d(v), 0., s, nu, th)[0]
    return quad(lambda v: (h(v) - f(v)) ** 2 / f(v), -np.inf, np.inf, limit=400)[0]

def jump_mix(y, loc, sc, nu, side="down"):
    f = tpdf(y - loc, sc, nu); best = None
    for th in GRID:
        h = (hminus if side == "down" else hplus)(y, loc, sc, nu, th)
        nll = lambda e: -np.mean(np.log((1 - e) * f + e * h))
        r = minimize_scalar(nll, bounds=(0., 0.5), method="bounded")
        lr = 2 * len(y) * (-r.fun - np.mean(np.log(f)))
        if best is None or lr > best["LR"]: best = dict(eps=float(r.x), theta=float(th), LR=float(lr))
    h = (hminus if side == "down" else hplus)(y, loc, sc, nu, best["theta"])
    e = best["eps"]; post = e * h / ((1 - e) * f + e * h)
    c2 = chi2_jump(nu, best["theta"]); T = len(y)
    best.update(chi2=c2, neff=T * e * e * c2, exp_jumps=float(post.sum()), mean_jump_sd=2 * best["theta"])
    best["zone"] = "negligible" if best["neff"] < 1 else ("grey" if best["neff"] < 100 else "usable")
    return best, post

# ---------------------------------------------------------------- main
def run(name, path, freq, session, out, drop_zeros=False):
    t0 = time.time(); rng = np.random.default_rng(20261031)
    r = returns(load(path), freq, session); z0 = float(np.mean(r.values == 0))
    if drop_zeros: r = r[r.values != 0]   # thin feeds (e.g. IEX-only): a zero bin means no trade, not a zero price change
    x = deseason(r); y = x.values
    res = dict(name=name, file=os.path.basename(path), freq_min=freq, start=str(x.index[0]), end=str(x.index[-1]), T=len(y),
               zero_share=z0, zeros_dropped=drop_zeros)
    m = fit_mix(y); vr, post_v = regime_index(y, m); res["vol_regime"] = {**vr, **{k: m[k] for k in ("pi0", "s0", "s1", "nu0", "nu1")}}
    g, sc = garch_fit(y); res["garch"] = g
    z = (y - g["mu"]) / (sc * np.sqrt(g["nu"] / (g["nu"] - 2)))
    mz = fit_mix(z); tr, _ = regime_index(z, mz); res["tail_regime"] = {**tr, **{k: mz[k] for k in ("pi0", "s0", "s1", "nu0", "nu1")}}
    sj, _ = srj(y, g["mu"], sc, g["nu"], rng); res["srj"] = sj
    jd, post_j = jump_mix(y, g["mu"], sc, g["nu"], "down"); ju, _ = jump_mix(y, g["mu"], sc, g["nu"], "up")
    res["jump_down"] = jd; res["jump_up"] = ju
    # monthly indices (full-sample GARCH and deseasonalisation; mixture refitted per month)
    rows = []
    for per, idx in pd.Series(np.arange(len(y)), index=x.index).groupby(x.index.to_period("M")):
        i = idx.values
        if len(i) < 400: continue
        yy, zz = y[i], z[i]
        try:
            mv = fit_mix(yy, [m["q"]]); v, _ = regime_index(yy, mv)
            mt = fit_mix(zz, [mz["q"]]); tt, _ = regime_index(zz, mt)
        except Exception: continue
        s, _ = srj(yy, g["mu"], sc[i], g["nu"], rng, 5000)
        j, _ = jump_mix(yy, g["mu"], sc[i], g["nu"], "down")
        rows.append(dict(month=str(per), T=len(i), vol_kappa=v["kappa"], vol_neff=v["neff"], vol_Z=v["Z"], vol_rho=v["rho_hat"],
                         tail_kappa=tt["kappa"], tail_neff=tt["neff"], tail_Z=tt["Z"], srj_down=s["srj_down"], p_down=s["p_down"],
                         srj_up=s["srj_up"], jump_eps=j["eps"], jump_theta=j["theta"], jump_neff=j["neff"], post_turb=float(post_v[i].mean()),
                         exp_jumps=float(post_j[i].sum())))
    mon = pd.DataFrame(rows); os.makedirs(out, exist_ok=True)
    mon.to_csv(os.path.join(out, f"{name}_monthly.csv"), index=False)
    res["seconds"] = time.time() - t0
    json.dump(res, open(os.path.join(out, f"{name}_index.json"), "w"), indent=1, default=float)
    return res, mon

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("file")
    ap.add_argument("--freq", type=int, default=5); ap.add_argument("--session", default=None); ap.add_argument("--out", default="out"); ap.add_argument("--drop-zeros", action="store_true")
    a = ap.parse_args(); res, mon = run(a.name, a.file, a.freq, a.session, a.out, a.drop_zeros)
    print(json.dumps({k: res[k] for k in ("name", "T", "start", "end")}, default=str))
    for k in ("vol_regime", "tail_regime", "jump_down", "jump_up", "srj"):
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in res[k].items() if kk != "Z_theta"})
    print(mon.round(3).tail(6).to_string())
