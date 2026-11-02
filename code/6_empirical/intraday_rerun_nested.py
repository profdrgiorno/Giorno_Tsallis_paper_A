#!/usr/bin/env python3
"""Rerun one intraday cell of Paper A (Section 5) with the nested G1 start for G2.
Identical to GT_GARCH_HMM_Joint_Experiment.py main() except for the extra G2 start, and it reports whether
loglik(G2) >= loglik(G1) before and after the fix.  Run once per market/frequency, e.g.
  python intraday_rerun_nested.py --repo PATH/Giorno_garch_hmm6 --csv WIN.csv --format win --market WIN --timeframe 1 --out WIN_M1
  python intraday_rerun_nested.py --repo PATH --csv US.csv --format us --market US --timeframe 15 --out US_M15
Then: python forecast_tests.py --file WIN:M1=WIN_M1/holdout_forecasts.csv ... (all eight cells)."""
import argparse, json
from pathlib import Path
import numpy as np
import sys, os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from nested_start import load_repo, nested_refit
ap=argparse.ArgumentParser()
ap.add_argument('--repo',default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'known_generator')); ap.add_argument('--csv'); ap.add_argument('--format',choices=['win','us'],default='win')
ap.add_argument('--market',default='WIN'); ap.add_argument('--timeframe',type=int,default=15)
ap.add_argument('--rth-start'); ap.add_argument('--rth-end'); ap.add_argument('--train-frac',type=float,default=.80)
ap.add_argument('--starts',type=int,default=3); ap.add_argument('--maxiter',type=int,default=220); ap.add_argument('--hac-lags',type=int,default=20)
ap.add_argument('--bootstrap',type=int,default=400); ap.add_argument('--seed',type=int,default=20261027); ap.add_argument('--out',default='RERUN')
ap.add_argument('--self-test',action='store_true')
a=ap.parse_args(); G=load_repo(a.repo); dest=Path(a.out); dest.mkdir(parents=True,exist_ok=True)
if a.self_test: ret,audit=G.simulate_example(n=1600,seed=a.seed)
else:
    start=a.rth_start or ('09:00' if a.format=='win' else '09:30'); end=a.rth_end or ('17:00' if a.format=='win' else '16:00')
    ret,audit=G.load_returns(a.csv,a.format,a.timeframe,start,end)
days=ret.timestamp.dt.normalize(); uniq=np.sort(days.unique()); cutoff=max(1,min(len(uniq)-1,int(np.floor(a.train_frac*len(uniq)))))
split=int(np.flatnonzero(days>=uniq[cutoff])[0])
train=ret.return_pct_log.to_numpy(np.float64)[:split]; mu=float(train.mean())
x=np.ascontiguousarray(train-mu); st=np.ascontiguousarray(ret.segment_start.iloc[:split].to_numpy(np.bool_))
m0=G.fit_model(x,st,hmm=False,nstarts=a.starts,maxiter=a.maxiter,seed=a.seed)
m1=G.fit_iid_mixture(x,st,nstarts=a.starts,maxiter=a.maxiter,seed=a.seed)
m2=G.fit_model(x,st,hmm=True,nstarts=a.starts,maxiter=a.maxiter,seed=a.seed)
before=m2['loglik_train']-m1['loglik_train']
m2=nested_refit(G,x,st,m1,m2,a.maxiter)
print(f"G2-G1 training loglik: original starts {before:.4f}; after nested start {m2['G2_minus_G1_loglik']:.4f}; rho={m2['p00']+m2['p11']-1:.3f}")
models={'garch_t':m0,'garch_mix_t':m1,'garch_hmm':m2}
G.evaluate(ret,mu,models,split,dest,a)
(dest/'run_metadata.json').write_text(json.dumps({'market':a.market,'timeframe':a.timeframe,'audit':audit,'train_n':split,'test_n':len(ret)-split,
    'training_mean':mu,'G2_minus_G1_loglik_original_starts':before,'models':models},indent=2,default=str))
