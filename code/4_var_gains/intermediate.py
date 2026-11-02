"""Intermediate-information designs for Paper A (T*kappa^2 between 1 and 100).
Two-state Student-t HMM, nu0=nu1=8, p00=0.975, p11=0.85 (pi0=0.857, rho=0.825), scale ratio r chosen so that
T*kappa^2 = 2, 8, 25, 60 at T=4,800.  Per path: fit static t, GARCH(1,1)-t and HMM-6 on the first 4,800 returns,
one-step VaR on the 1,200-return holdout (95/99/99.5%), pinball loss and breaches; also rho_hat and the Z test.
Usage: python intermediate.py NPATHS PART NPARTS OUT.json"""
import json, os, sys, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.environ.get("JUMPARM_DIR", HERE))
import jumparm as J
from scipy.stats import t as tdist
from scipy.integrate import quad

DESIGNS = {"inter_2": 1.3535, "inter_8": 1.5147, "inter_25": 1.7163, "inter_60": 1.9470}
P00, P11, NU, S0, N, NTR = .975, .85, 8., .7, 6000, 4800

def simulate(r, seed, burn=600):
    rng = np.random.default_rng(seed); total = N + burn; u = rng.random(total); s = np.empty(total, np.int8)
    pi0 = (1 - P11) / (2 - P00 - P11); s[0] = 0 if u[0] < pi0 else 1
    for t in range(1, total): s[t] = 0 if u[t] < (P00 if s[t - 1] == 0 else 1 - P11) else 1
    z = rng.standard_t(NU, total) * np.where(s == 0, S0, S0 * r)
    return z[burn:], s[burn:]

def kappa_true(r):
    p0 = (1 - P11) / (2 - P00 - P11); p1 = 1 - p0
    f0 = lambda y: tdist.pdf(y / S0, NU) / S0; f1 = lambda y: tdist.pdf(y / (S0 * r), NU) / (S0 * r)
    fb = lambda y: p0 * f0(y) + p1 * f1(y)
    return p0 * p1 * quad(lambda y: (f0(y) - f1(y)) ** 2 / fb(y), -np.inf, np.inf, limit=300)[0]

def z_test(y):
    """lag-one contrast autocorrelation test from an i.i.d. two-component t mixture (Paper A, Section 3.3)"""
    from scipy.optimize import minimize
    from scipy.special import expit, logit
    def unpack(q): return expit(q[0]), np.exp(q[1]), np.exp(q[2]), 2.05 + 97.95 * expit(q[3]), 2.05 + 97.95 * expit(q[4])
    def nll(q):
        p0, s0, s1, n0, n1 = unpack(q)
        return -np.mean(np.logaddexp(np.log(p0) + tdist.logpdf(y / s0, n0) - np.log(s0), np.log1p(-p0) + tdist.logpdf(y / s1, n1) - np.log(s1)))
    sd = y.std(); best = None
    for q0 in ([logit(.8), np.log(.8 * sd), np.log(1.5 * sd), logit(6 / 97.95), logit(6 / 97.95)],
               [logit(.6), np.log(.9 * sd), np.log(1.2 * sd), logit(18 / 97.95), logit(4 / 97.95)]):
        r = minimize(nll, np.array(q0), method="L-BFGS-B")
        if best is None or r.fun < best.fun: best = r
    p0, s0, s1, n0, n1 = unpack(best.x)
    f0 = tdist.pdf(y / s0, n0) / s0; f1 = tdist.pdf(y / s1, n1) / s1; fb = p0 * f0 + (1 - p0) * f1
    d = (f0 - f1) / fb; d = d - d.mean(); a1 = np.dot(d[:-1], d[1:]) / np.dot(d, d)
    return float(np.sqrt(len(y) - 1) * a1)

if __name__ == "__main__":
    npaths, part, nparts, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    recs = []; t0 = time.time(); k = 0
    for name, r in DESIGNS.items():
        p = (P00, P11, NU, NU, S0, S0 * r); kap = kappa_true(r)
        for rep in range(npaths):
            k += 1
            if k % nparts != part: continue
            seed = [20261101, list(DESIGNS).index(name), rep]
            y, _ = simulate(r, seed); tr = y[:NTR]
            mods = {"t": J.fit_t(tr), "garch": J.fit_ap(tr, "garch"), "hmm": J.fit_hmm(tr, p)}
            b = mods["hmm"]["b"]
            rho_hat = b[0] + b[1] - 1   # invariant to label switching
            tv = J.true_var(y, p, 0.0, 0.0, NTR)
            fc = J.forecasts(y, NTR, {**mods, "aparch": mods["garch"]}, p); fc.pop("APARCH-t")
            recs.append(dict(design=name, ratio=r, kappa=kap, neff=NTR * kap ** 2, rep=rep, rho_hat=float(rho_hat),
                             Z=z_test(tr), hmm_b=b.tolist(), eval=J.evaluate(y, NTR, fc, tv)))
            if len(recs) % 5 == 0:
                json.dump(recs, open(out, "w")); print(len(recs), round(time.time() - t0), "s", flush=True)
    json.dump(recs, open(out, "w")); print("done", len(recs), round(time.time() - t0), flush=True)
