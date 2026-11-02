#!/usr/bin/env python3
"""Forecast-comparison tests for Paper A on holdout_forecasts.csv files written by
GT_GARCH_HMM_Joint_Experiment.py (or daily_pipeline.py).

For every file and every score (log score; pinball loss at 95, 99, 99.5% VaR):
  * Diebold-Mariano test for each model pair (Newey-West HAC variance);
  * Giacomini-White (2006) conditional test with instruments (1, lagged loss differential);
  * Model confidence set (Hansen, Lunde and Nason 2011), T_max statistic, stationary bootstrap.
Usage:
  python forecast_tests.py --file US:M1=path/holdout_forecasts.csv --file WIN:M15=... --out results_dir
"""
import argparse, itertools, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm, chi2
MODELS=('garch_t','garch_mix_t','garch_hmm'); TAGS=('95p0','99p0','99p5')
def nw_var(x,L):
    x=x-x.mean(); n=len(x); g=x@x/n
    for l in range(1,L+1): g+=2*(1-l/(L+1))*(x[l:]@x[:-l])/n
    return g
def losses(df,score):
    if score=='log': return {m:-df[f'{m}_logdensity'].to_numpy(float) for m in MODELS}
    return {m:df[f'{m}_loss_{score}'].to_numpy(float) for m in MODELS}
def dm(l1,l2,L):
    d=l1-l2; v=nw_var(d,L); z=d.mean()/np.sqrt(v/len(d)); return d.mean(),z,2*norm.sf(abs(z))
def gw(l1,l2):
    d=l1-l2; h=np.column_stack([np.ones(len(d)-1),d[:-1]]); z=h*d[1:,None]; zb=z.mean(0)
    Om=(z-zb).T@(z-zb)/len(z); stat=len(z)*zb@np.linalg.solve(Om,zb); return stat,chi2.sf(stat,h.shape[1])
def stat_boot_idx(n,b,rng):
    idx=np.empty(n,int); idx[0]=rng.integers(n); p=1/b
    for t in range(1,n): idx[t]=rng.integers(n) if rng.random()<p else (idx[t-1]+1)%n
    return idx
def mcs(L,names,B=2000,block=10,alpha=0.10,seed=1):
    rng=np.random.default_rng(seed); n,m=L.shape; alive=list(range(m)); pvals={}; pmax=0.0
    boots=[stat_boot_idx(n,block,rng) for _ in range(B)]
    while len(alive)>1:
        Lm=L[:,alive]   # d_i = mean over j of (L_i - L_j), Hansen-Lunde-Nason T_max
        di=(Lm-Lm.mean(1,keepdims=True)).mean(0)*len(alive)/(len(alive)-1)
        bs=np.array([((Lm[ix]-Lm[ix].mean(1,keepdims=True)).mean(0)*len(alive)/(len(alive)-1)) for ix in boots])
        se=bs.std(0); t=di/se; Tmax=t.max(); Tb=((bs-di)/se).max(1); p=float((Tb>=Tmax).mean())
        pmax=max(pmax,p); worst=alive[int(np.argmax(t))]; pvals[names[worst]]=pmax; alive.remove(worst)
    pvals[names[alive[0]]]=1.0
    return pvals
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--file',action='append',required=True,help='LABEL=path')
    ap.add_argument('--hac-lags',type=int,default=20); ap.add_argument('--B',type=int,default=2000); ap.add_argument('--block',type=float,default=10.)
    ap.add_argument('--out',default='forecast_tests_out'); a=ap.parse_args()
    out=Path(a.out); out.mkdir(exist_ok=True); rows=[]; mrows=[]
    for spec in a.file:
        lab,path=spec.split('=',1); df=pd.read_csv(path)
        for score in ('log',)+TAGS:
            Ls=losses(df,score)
            for m1,m2 in itertools.combinations(MODELS,2):
                md,z,p=dm(Ls[m1],Ls[m2],a.hac_lags); s,pg=gw(Ls[m1],Ls[m2])
                rows.append(dict(cell=lab,score=score,pair=f'{m1} vs {m2}',mean_loss_diff=md,dm_z=z,dm_p=p,gw_stat=s,gw_p=pg,n=len(df)))
            pv=mcs(np.column_stack([Ls[m] for m in MODELS]),MODELS,B=a.B,block=a.block)
            for m,p in pv.items(): mrows.append(dict(cell=lab,score=score,model=m,mcs_p=p,in_mcs90=p>=0.10,in_mcs75=p>=0.25))
    r=pd.DataFrame(rows); m=pd.DataFrame(mrows)
    r.to_csv(out/'dm_gw_tests.csv',index=False); m.to_csv(out/'mcs.csv',index=False)
    print(r.round(4).to_string(index=False)); print(m.round(3).to_string(index=False))
    print('\nNote: positive mean_loss_diff means the first model has the larger loss. DM uses HAC; GW instruments are (1, lagged differential). MCS p-values follow Hansen, Lunde and Nason (2011).')
