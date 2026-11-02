"""Reproduce Table 2 of Paper A v2 (VaR gain of the HMM vs T*kappa^2).

Inputs:  pilot_var_eval.csv  (per-path VaR evaluation, jump-free arm, from jumparm.py)
         kappa_by_design.csv (exact kappa and n_eff per design, from theorem_check2.py)
Output:  var_by_neff.csv and t5.tex
Usage:   python var_by_neff.py pilot_var_eval.csv kappa_by_design.csv
"""
import sys
import pandas as pd

ev = pd.read_csv(sys.argv[1]); kap = pd.read_csv(sys.argv[2])
ev = ev[ev.arm == "nojump"]
neff = dict(zip(kap.design, kap.n_eff)); neff["single_student_t"] = 0.0
names = {"calibration_anchor": "Calibration anchor", "full_hmm6": "Scale and tail",
         "scale_only": "Scale only", "tail_only": "Tail only", "weak_tail": "Weak tail",
         "weak_persistence": "Weak persistence", "iid_mixture": "Independent mixture",
         "single_student_t": "Identical regimes"}
m = ev.groupby(["scenario", "level", "model"]).agg(pin=("pinball", "mean"), br=("breach_rate", "mean")).reset_index()
rows = []
for (s, lv), g in m.groupby(["scenario", "level"]):
    L = dict(zip(g.model, g.pin)); B = dict(zip(g.model, g.br))
    hmm = [k for k in L if "HMM" in k][0]; gar = [k for k in L if "GARCH" in k][0]
    rows.append(dict(design=s, neff=neff.get(s, float("nan")), level=lv,
                     gain_vs_garch=100 * (L[gar] - L[hmm]) / L[gar],
                     gain_vs_t=100 * (L["Student-t"] - L[hmm]) / L["Student-t"],
                     br_hmm=100 * B[hmm]))
out = pd.DataFrame(rows); out.to_csv("var_by_neff.csv", index=False)

def f(x):
    s = f"{x:.1f}"; return s.replace("-", "$-$") if s != "-0.0" else "0.0"
lines = []
for s in names:
    o = out[out.design == s].set_index("level")
    n = o.neff.iloc[0]; ns = "$<$0.01" if n < 0.01 else (f"{n:.2f}" if n < 1 else f"{n:.0f}")
    lines.append(f"{names[s]} & {ns} & {f(o.gain_vs_t[0.05])} & {f(o.gain_vs_t[0.01])} & "
                 f"{f(o.gain_vs_t[0.005])} & {f(o.gain_vs_garch[0.01])} & {o.br_hmm[0.01]:.2f}\\\\")
open("t5.tex", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
