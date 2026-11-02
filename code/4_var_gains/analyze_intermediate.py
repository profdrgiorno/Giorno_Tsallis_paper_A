"""Summaries of the intermediate designs: VaR gains (with path-bootstrap 95% intervals), rho recovery, Z test power."""
import glob, json, sys, numpy as np, pandas as pd
recs = sum((json.load(open(f)) for f in sorted(glob.glob("inter_p*.json"))), [])
rows = []
for r in recs:
    for e in r["eval"]:
        rows.append(dict(design=r["design"], neff=r["neff"], kappa=r["kappa"], rep=r["rep"], model=e["model"], level=e["level"],
                         pin=e["pinball"], br=e["breach_rate"]))
d = pd.DataFrame(rows); rng = np.random.default_rng(1)
out = []
for des, g in d.groupby("design"):
    P = g.pivot_table(index=["rep", "level"], columns="model", values="pin").reset_index()
    rec = dict(design=des, neff=g.neff.iloc[0], kappa=g.kappa.iloc[0], paths=g.rep.nunique())
    for lv in (0.05, 0.01, 0.005):
        q = P[P.level == lv]
        for bench, tag in (("Student-t", "t"), ("GARCH-t", "g")):
            gain = 100 * (q[bench].mean() - q["HMM-6"].mean()) / q[bench].mean()
            bs = []
            idx = np.arange(len(q))
            for _ in range(2000):
                s = q.iloc[rng.choice(idx, len(idx))]; bs.append(100 * (s[bench].mean() - s["HMM-6"].mean()) / s[bench].mean())
            rec[f"gain_{tag}_{lv}"] = gain; rec[f"lo_{tag}_{lv}"] = np.quantile(bs, .025); rec[f"hi_{tag}_{lv}"] = np.quantile(bs, .975)
    rec["br_hmm_99"] = 100 * g[(g.model == "HMM-6") & (g.level == 0.01)].br.mean()
    out.append(rec)
S = pd.DataFrame(out).sort_values("neff")
R = pd.DataFrame([dict(design=r["design"], rho_hat=r["rho_hat"], Z=r["Z"]) for r in recs]).groupby("design").agg(
    rho_mean=("rho_hat", "mean"), rho_sd=("rho_hat", "std"), power=("Z", lambda z: float((np.abs(z) > 1.96).mean())))
S = S.merge(R, on="design"); S.to_csv("intermediate_summary.csv", index=False)
pd.set_option("display.width", 250); print(S.round(3).T.to_string())
