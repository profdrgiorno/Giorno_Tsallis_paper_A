"""Parametric-bootstrap version of the regime test after GARCH filtering (Paper A, Section 3.3).

Statistic: Z = sqrt(T-1) * acf1(d_hat) computed on GARCH(1,1)-t standardised returns, d_hat from an i.i.d.
two-component Student-t mixture fitted to those residuals.
Bootstrap null (no persistence): GARCH(1,1) with i.i.d. innovations drawn from the fitted mixture (rescaled to unit
variance); GARCH and mixture are re-estimated on every bootstrap sample.
Size/power by the warp-speed method (Giacomini, Politis and White, 2013): one bootstrap draw per Monte Carlo path;
reject when |Z| exceeds the 95% quantile of the pooled |Z*|.
Usage: python boot_test.py DGP NREP T SEED OUT.csv     DGP in {garch_t, garch_mix_iid, garch_hmm_tail, garch_hmm_scale}
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.environ.get("INDEX_DIR", os.path.dirname(os.path.abspath(__file__))))
import monitor as M
from scipy.stats import t as tdist

OM, AL, BE = .02, .07, .91
DGPS = {  # (p00, p11, nu0, nu1, scale0, scale1)
    "garch_t":         (1.0, 0.0, 7.5, 7.5, 1.0, 1.0),
    "garch_mix_iid":   (.82, .18, 24., 4.6, 1.0, 1.0),     # two tail regimes, no persistence (rho = 0)
    "garch_hmm_tail":  (.975, .85, 24., 4.6, 1.0, 1.0),    # persistent tail-only regimes (rho = 0.825)
    "garch_hmm_scale": (.975, .85, 8., 8., 1.0, 1.7),      # persistent scale regimes (rho = 0.825)
}

def regimes(p00, p11, T, rng):
    if p00 == 1.0: return np.zeros(T, np.int8)
    s = np.empty(T, np.int8); pi0 = (1 - p11) / (2 - p00 - p11); u = rng.random(T); s[0] = 0 if u[0] < pi0 else 1
    for t in range(1, T): s[t] = 0 if u[t] < (p00 if s[t - 1] == 0 else 1 - p11) else 1
    return s

def innovations(spec, T, rng, s=None):
    p00, p11, n0, n1, a0, a1 = spec
    if s is None: s = regimes(p00, p11, T, rng)
    # unit-variance t draws times regime scale, then normalise to unit variance
    z = np.where(s == 0, a0 * rng.standard_t(n0, T) * np.sqrt((n0 - 2) / n0), a1 * rng.standard_t(n1, T) * np.sqrt((n1 - 2) / n1))
    pi0 = 1.0 if p00 == 1.0 else (1 - p11) / (2 - p00 - p11)
    return z / np.sqrt(pi0 * a0 ** 2 + (1 - pi0) * a1 ** 2)

def garch_path(z, om=OM, al=AL, be=BE, burn=500):
    T = len(z); h = om / (1 - al - be); y = np.empty(T); e = 0.
    for t in range(T):
        h = om + al * e * e + be * h; e = np.sqrt(h) * z[t]; y[t] = e
    return y[burn:]

def statistic(y):
    g, sc = M.garch_fit(y); z = (y - g["mu"]) / (sc * np.sqrt(g["nu"] / (g["nu"] - 2)))
    m = M.fit_mix(z); r, _ = M.regime_index(z, m)
    return r["Z"], r["neff"], g, m

def mix_draw(m, T, rng):
    k = rng.random(T) < m["pi0"]
    z = np.where(k, m["s0"] * rng.standard_t(m["nu0"], T), m["s1"] * rng.standard_t(m["nu1"], T))
    v = m["pi0"] * m["s0"] ** 2 * m["nu0"] / (m["nu0"] - 2) + (1 - m["pi0"]) * m["s1"] ** 2 * m["nu1"] / (m["nu1"] - 2)
    return z / np.sqrt(v)

if __name__ == "__main__":
    dgp, nrep, T, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    rng = np.random.default_rng([seed, list(DGPS).index(dgp)]); rows = []
    for i in range(nrep):
        y = garch_path(innovations(DGPS[dgp], T + 500, rng))
        Z, neff, g, m = statistic(y)
        # one bootstrap draw under the no-persistence null, from the fitted model
        zb = mix_draw(m, T + 500, rng); yb = garch_path(zb, g["omega"], g["alpha"], g["beta"])
        Zb, neffb, _, _ = statistic(yb)
        rows.append(dict(dgp=dgp, rep=i, T=T, Z=Z, neff=neff, Zb=Zb, neffb=neffb))
        if (i + 1) % 25 == 0: pd.DataFrame(rows).to_csv(out, index=False); print(i + 1, flush=True)
    pd.DataFrame(rows).to_csv(out, index=False); print("done", flush=True)
