#!/usr/bin/env python3
"""Persistence diagnostic of Paper A (Theorem 2) for any return series.

1. Optionally standardise returns with a GARCH(1,1)-t filter (--garch).
2. Fit an i.i.d. two-component Student-t mixture by maximum likelihood.
3. Report the regime contrast kappa = pi0*pi1*int (f0-f1)^2/fbar (also the order-2 Tsallis mutual information
   between regime and observation), the effective persistence sample size n_eff = T*kappa^2, the persistence
   detectable with 80% power by the score test, and the score test Z = sqrt(T-1)*acf1(d-hat) of rho = 0.
4. Parametric-bootstrap p-value of Z (i.i.d. data only; with --garch the asymptotic N(0,1) p-value is indicative).
Usage:
  python persistence_check.py --csv prices.csv --date-col Date --price-col Close [--start 1990-01-01] [--garch] [--B 199]
  MetaTrader exports: --mt5 (tab-separated with <DATE>, <CLOSE>)."""
import argparse, json, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, gammaln
from scipy.integrate import quad
from scipy.stats import norm
def lt(y,nu,s): return gammaln((nu+1)/2)-gammaln(nu/2)-.5*np.log(np.pi*nu)-np.log(s)-.5*(nu+1)*np.log1p((y/s)**2/nu)
def fit_mix(y):
    tr=lambda q:(expit(q[0]),2.05+97.95*expit(q[1]),2.05+97.95*expit(q[2]),np.exp(q[3]),np.exp(q[4]))
    f=lambda q:(lambda w,n0,n1,s0,s1:-np.mean(np.logaddexp(np.log(w)+lt(y,n0,s0),np.log1p(-w)+lt(y,n1,s1))))(*tr(q))
    sd=y.std(); best=None
    for st in ([.5,8,8,.7*sd,1.4*sd],[.2,3,25,.8*sd,.9*sd],[.7,20,5,.8*sd,1.2*sd]):
        r=minimize(f,[logit(st[0]),logit((st[1]-2.05)/97.95),logit((st[2]-2.05)/97.95),np.log(st[3]),np.log(st[4])],method='L-BFGS-B',options={'maxiter':800})
        if best is None or r.fun<best.fun: best=r
    return tr(best.x)
def contrast(y,w,n0,n1,s0,s1):
    f0=np.exp(lt(y,n0,s0)); f1=np.exp(lt(y,n1,s1)); return (f0-f1)/(w*f0+(1-w)*f1)
def kappa(w,n0,n1,s0,s1):
    f0=lambda x: np.exp(lt(x,n0,s0)); f1=lambda x: np.exp(lt(x,n1,s1))
    return w*(1-w)*quad(lambda x:(f0(x)-f1(x))**2/(w*f0(x)+(1-w)*f1(x)),-np.inf,np.inf,limit=500)[0]
def zstat(y,par):
    d=contrast(y,*par); dc=d-d.mean(); return np.sqrt(len(y)-1)*(dc[:-1]*dc[1:]).mean()/dc.var()
def garch_standardise(r):
    """GARCH(1,1) with unit-variance Student-t innovations, constant mean; returns standardised residuals."""
    x0=r-r.mean(); v=x0.var()
    def unpack(q): return q[0],np.exp(q[1]),.999*expit(q[2])*expit(q[3]),.999*expit(q[2])*(1-expit(q[3])),2.05+97.95*expit(q[4])
    def path(q):
        mu,om_v,a,b,nu=unpack(q); x=r-mu; h=np.empty(len(x)); h[0]=v; om=om_v*(1-a-b)
        for t in range(1,len(x)): h[t]=om+a*x[t-1]**2+b*h[t-1]
        return x,h,nu
    def nll(q):
        x,h,nu=path(q); s=np.sqrt(h*(nu-2)/nu); return -np.mean(lt(x,nu,s))
    r_=minimize(nll,[r.mean(),np.log(v),logit(.97/.999),logit(.07),logit((6-2.05)/97.95)],method='L-BFGS-B',options={'maxiter':800})
    x,h,nu=path(r_.x); return x/np.sqrt(h)
def simulate_mix(par,T,rng):
    w,n0,n1,s0,s1=par; k=rng.random(T)<w
    return np.where(k,rng.standard_t(n0,T)*s0,rng.standard_t(n1,T)*s1)
def run(y,B=199,seed=1,garch_used=False):
    par=fit_mix(y); kap=kappa(*par); T=len(y); Z=zstat(y,par)
    out=dict(T=T,weight0=par[0],nu0=par[1],nu1=par[2],s0=par[3],s1=par[4],kappa=kap,tsallis2_mutual_information=kap,
             n_eff=T*kap**2,rho_detectable_80pct=(1.96+0.842)/(kap*np.sqrt(T)) if kap>0 else np.inf,
             Z=Z,p_asymptotic=2*norm.sf(abs(Z)))
    if B>0 and not garch_used:
        rng=np.random.default_rng(seed); zb=[]
        for _ in range(B):
            yb=simulate_mix(par,T,rng); zb.append(zstat(yb,fit_mix(yb)))
        zb=np.array(zb); out['p_bootstrap']=float((1+np.sum(np.abs(zb)>=abs(Z)))/(B+1)); out['B']=B
    return out
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); ap.add_argument('--mt5',action='store_true')
    ap.add_argument('--date-col',default='Date'); ap.add_argument('--price-col',default='Close'); ap.add_argument('--start'); ap.add_argument('--end')
    ap.add_argument('--garch',action='store_true'); ap.add_argument('--B',type=int,default=199); ap.add_argument('--out',default='persistence_check.json')
    a=ap.parse_args()
    if a.mt5:
        d=pd.read_csv(a.csv,sep='\t'); d['Date']=pd.to_datetime(d['<DATE>'].astype(str),format='%Y.%m.%d'); d['Close']=d['<CLOSE>']; dc,pc='Date','Close'
    else:
        d=pd.read_csv(a.csv); dc,pc=a.date_col,a.price_col; d[dc]=pd.to_datetime(d[dc])
    d=d.sort_values(dc)
    if a.start: d=d[d[dc]>=a.start]
    if a.end: d=d[d[dc]<=a.end]
    r=100*np.log(d[pc].astype(float)).diff().dropna().to_numpy()
    y=garch_standardise(r) if a.garch else r-r.mean()
    res=run(y,a.B,garch_used=a.garch); res['garch_filter']=a.garch
    print(json.dumps(res,indent=1,default=float)); json.dump(res,open(a.out,'w'),indent=1,default=float)
