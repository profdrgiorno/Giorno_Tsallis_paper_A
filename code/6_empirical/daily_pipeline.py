#!/usr/bin/env python3
"""Paper A daily analyses (Section 7) for any daily OHLC file, e.g. WIN$ futures or SPY.
  A. GARCH-t (G0), GARCH-i.i.d. mixture (G1), GARCH-HMM (G2) with the nested G1 start; holdout density/VaR tests.
  B. Persistence diagnostic on the G1 standardised innovations: kappa, n_eff, contrast-autocorrelation Z.
  C. GJR-GARCH robustness of the contrast autocorrelation.
  D. Marked Student-t HMM versus static Student-t (marks: weekend/holiday closures, or overnight with --ohlc-split),
     VaR evaluation and tail-equality likelihood ratio with parametric bootstrap.
Usage:
  python daily_pipeline.py --repo PATH/Giorno_garch_hmm6 --csv WIN_D1.csv --mt5 --start 2005-01-01 --out WIN_D1 --B 399
  python daily_pipeline.py --repo PATH --csv SPY.csv --date-col Date --start 1995-01-01 --out SPY_D1 --B 399 --ohlc-split
Compile first: gcc -O2 -shared -fPIC -o mhmm.so mhmm.c -lm   (Windows: build mhmm.dll with MinGW)."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import gammaln, expit
from scipy.stats import norm, chi2, t as tdist
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from nested_start import load_repo, nested_refit
import marked_core as MC
def load_ohlc(a):
    if a.mt5:
        d=pd.read_csv(a.csv,sep='\t'); d=pd.DataFrame({'Date':pd.to_datetime(d['<DATE>'].astype(str),format='%Y.%m.%d'),
              'Open':d['<OPEN>'],'High':d['<HIGH>'],'Low':d['<LOW>'],'Close':d['<CLOSE>']})
    else:
        d=pd.read_csv(a.csv); d=d.rename(columns={a.date_col:'Date'}); d['Date']=pd.to_datetime(d['Date'])
    d=d.sort_values('Date').drop_duplicates('Date').reset_index(drop=True)
    if a.start: d=d[d.Date>=a.start]
    if a.end: d=d[d.Date<=a.end]
    return d.reset_index(drop=True)
def ft(z,nu): s2=(nu-2)/nu; return np.exp(gammaln((nu+1)/2)-gammaln(nu/2)-0.5*np.log(np.pi*nu*s2)-0.5*(nu+1)*np.log1p(z*z/(s2*nu)))
def contrast_stats(z,nu0,nu1,w):
    fb=lambda u: w*ft(u,nu0)+(1-w)*ft(u,nu1)
    chi=quad(lambda u:(ft(u,nu0)-ft(u,nu1))**2/fb(u),-np.inf,np.inf,limit=500)[0]; kap=w*(1-w)*chi
    d=(ft(z,nu0)-ft(z,nu1))/fb(z); dc=d-d.mean(); ac=float((dc[:-1]*dc[1:]).mean()/dc.var()); T=len(z)
    zz=z**2-np.mean(z**2); acz=float((zz[:-1]*zz[1:]).mean()/zz.var())
    return dict(kappa=kap,n_eff=T*kap**2,acf1_d=ac,Z=np.sqrt(T-1)*ac,p=float(2*norm.sf(abs(np.sqrt(T-1)*ac))),
                rho_implied=ac/kap if kap>0 else np.nan,rho_detectable_80pct=2.802/(kap*np.sqrt(T)),acf1_z2=acz)
def gjr_check(x):
    def lt_(z2,nu,h): s2=h*(nu-2)/nu; return gammaln((nu+1)/2)-gammaln(nu/2)-0.5*np.log(np.pi*nu*s2)-0.5*(nu+1)*np.log1p(z2/(s2*nu))
    def unpack(p):
        v0=np.exp(p[0]); P=.998*expit(p[1]); u=expit(p[2]); v=expit(p[3])
        return v0,u*P*(1-v),2*u*P*v,P*(1-u),2.05+np.exp(p[4]),2.05+np.exp(p[5]),expit(p[6])
    def hp(om,a,g,b,v0):
        h=np.empty(len(x)); h[0]=v0
        for t in range(1,len(x)): h[t]=om+(a+g*(x[t-1]<0))*x[t-1]**2+b*h[t-1]
        return h
    def nll(p):
        v0,a,g,b,n0,n1,w=unpack(p); h=hp(v0*(1-a-.5*g-b),a,g,b,v0)
        return -np.logaddexp(np.log(w)+lt_(x*x,n0,h),np.log1p(-w)+lt_(x*x,n1,h)).sum()
    best=None
    for init in ([np.log(x.var()),5.0,-2.4,0.0,np.log(0.4),np.log(20),-1.4],[np.log(x.var()),4.0,-2.0,1.0,np.log(1.0),np.log(8),-0.5],[np.log(x.var()),5.5,-2.8,-1.0,np.log(0.3),np.log(30),-1.6]):
        r=minimize(nll,np.array(init),method='L-BFGS-B',options={'maxiter':600})
        if best is None or r.fun<best.fun: best=r
    v0,a,g,b,n0,n1,w=unpack(best.x)
    if n0>n1: n0,n1,w=n1,n0,1-w
    h=hp(v0*(1-a-.5*g-b),a,g,b,v0); out=contrast_stats(x/np.sqrt(h),n0,n1,w)
    out.update(loglik=-best.fun,alpha=a,gamma=g,beta=b,nu0=n0,nu1=n1,pi0=w); return out
def marked_block(y,o,split,B,seed):
    ytr,otr=y[:split],o[:split]
    pt,llt=MC.fit_t(ytr,otr); ph,llh,_=MC.fit_hmm(ytr,otr); pr,llr,_=MC.fit_hmm(ytr,otr,restricted=True)
    if ph[2]>ph[3]: ph=ph[[1,0,3,2,5,4,7,6]]
    pred=np.empty(len(y)); MC.ll_hmm(y,o,ph,pred); te=np.arange(split,len(y)); rows=[]
    for a_ in (0.05,0.01,0.005):
        qt=np.array([tdist.ppf(a_,pt[0])*pt[1]*(pt[2] if o[t] else 1) for t in te])
        qh=np.array([MC.mix_quantile(a_,pred[t],ph[2],ph[4]*(ph[6] if o[t] else 1),ph[3],ph[5]*(ph[7] if o[t] else 1)) for t in te])
        yt=y[te]; lt_=(a_-(yt<qt))*(yt-qt); lh=(a_-(yt<qh))*(yt-qh); diff=lt_-lh
        rows.append(dict(level=100*(1-a_),VaR_t=float(-qt.mean()),VaR_hmm=float(-qh.mean()),viol_t=float(100*(yt<qt).mean()),viol_hmm=float(100*(yt<qh).mean()),
            kupiec_p_t=MC.kupiec(int((yt<qt).sum()),len(te),a_),kupiec_p_hmm=MC.kupiec(int((yt<qh).sum()),len(te),a_),
            pinball_gain_pct=float(100*diff.mean()/lt_.mean()),hac_z=float(diff.mean()/MC.nw_se(diff,20))))
    LR=2*(llh-llr); rng=np.random.default_rng(seed); boot=[]
    p00,p11=pr[0],pr[1]; pi0=(1-p11)/(2-p00-p11)
    for b in range(B):
        k=np.empty(split,int); k[0]=rng.random()>pi0; u=rng.random(split)
        for t in range(1,split): k[t]=(u[t]>p00) if k[t-1]==0 else (u[t]>1-p11)
        sc=np.where(k==0,pr[4]*np.where(otr==1,pr[6],1),pr[5]*np.where(otr==1,pr[7],1)); yb=rng.standard_t(pr[2],split)*sc
        _,l1,_=MC.fit_hmm(yb,otr); _,l0,_=MC.fit_hmm(yb,otr,restricted=True); boot.append(max(2*(l1-l0),0.0))
    return dict(t_params=pt.tolist(),hmm_params=ph.tolist(),loglik_t=llt,loglik_hmm=llh,loglik_eqtail=llr,LR_tail=LR,chi2_p=float(chi2.sf(LR,1)),
                boot_p=float((1+np.sum(np.array(boot)>=LR))/(B+1)) if B>0 else None,B=B,var=rows)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'known_generator')); ap.add_argument('--csv',required=True); ap.add_argument('--mt5',action='store_true')
    ap.add_argument('--date-col',default='Date'); ap.add_argument('--start'); ap.add_argument('--end'); ap.add_argument('--out',default='DAILY_OUT')
    ap.add_argument('--B',type=int,default=399); ap.add_argument('--ohlc-split',action='store_true'); ap.add_argument('--train-frac',type=float,default=.8)
    ap.add_argument('--starts',type=int,default=3); ap.add_argument('--maxiter',type=int,default=220); ap.add_argument('--hac-lags',type=int,default=20)
    ap.add_argument('--bootstrap',type=int,default=400); ap.add_argument('--seed',type=int,default=20261027)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); G=load_repo(a.repo); d=load_ohlc(a)
    # ---- A
    r=100*np.log(d.Close/d.Close.shift(1)); ret=pd.DataFrame({'timestamp':d.Date,'return_pct_log':r}).iloc[1:].reset_index(drop=True)
    ret['segment_start']=False; ret.loc[0,'segment_start']=True; n=len(ret); split=int(np.floor(a.train_frac*n))
    tr=ret.return_pct_log.to_numpy(np.float64)[:split]; mu=float(tr.mean()); x=np.ascontiguousarray(tr-mu); st=np.ascontiguousarray(ret.segment_start.iloc[:split].to_numpy(np.bool_))
    m0=G.fit_model(x,st,hmm=False,nstarts=a.starts,maxiter=a.maxiter,seed=a.seed); m1=G.fit_iid_mixture(x,st,nstarts=a.starts,maxiter=a.maxiter,seed=a.seed)
    m2=G.fit_model(x,st,hmm=True,nstarts=a.starts,maxiter=a.maxiter,seed=a.seed); before=m2['loglik_train']-m1['loglik_train']; m2=nested_refit(G,x,st,m1,m2,a.maxiter)
    G.evaluate(ret,mu,{'garch_t':m0,'garch_mix_t':m1,'garch_hmm':m2},split,out,a)
    res={'sample':[str(ret.timestamp.iloc[0].date()),str(ret.timestamp.iloc[-1].date())],'train_n':split,'test_n':n-split,
         'A':{'loglik':{'G0':m0['loglik_train'],'G1':m1['loglik_train'],'G2':m2['loglik_train']},'G2_minus_G1_original_starts':before,
              'G2_rho':m2['p00']+m2['p11']-1,'G2_nu':[m2['nu0'],m2['nu1']],'G2_p':[m2['p00'],m2['p11']]}}
    # ---- B (G1 standardised innovations, estimation sample)
    om=m1['vbar']*(1-m1['alpha']-m1['beta']); h=np.empty(split); h[0]=m1['vbar']
    for t in range(1,split): h[t]=om+m1['alpha']*x[t-1]**2+m1['beta']*h[t-1]
    res['B']=contrast_stats(x/np.sqrt(h),m1['nu0'],m1['nu1'],m1['p00'])
    # ---- C
    res['C_gjr']=gjr_check(x)
    # ---- D
    if a.ohlc_split:
        dd=d.copy(); dd['pc']=dd.Close.shift(1); dd=dd.iloc[1:]; bad=np.isclose(dd.Open,dd.pc,atol=1e-12)|np.isclose(dd.Close,dd.Open,atol=1e-12); dd=dd[~bad]
        y=np.ravel(np.column_stack([100*np.log(dd.Open/dd.pc),100*np.log(dd.Close/dd.Open)])); o=np.ravel(np.column_stack([np.ones(len(dd),int),np.zeros(len(dd),int)]))
        splitD=2*int(np.floor(a.train_frac*len(dd))); res['D_mode']='overnight/intraday'; res['D_dropped_days']=int(bad.sum())
    else:
        y=ret.return_pct_log.to_numpy(); o=(ret.timestamp.diff().dt.days.fillna(1).to_numpy()>1).astype(int); splitD=split; res['D_mode']='weekend/holiday marks'
    res['D']=marked_block(y,o,splitD,a.B,a.seed)
    (out/'daily_results.json').write_text(json.dumps(res,indent=1,default=float)); print(json.dumps({k:res[k] for k in ('A','B')},indent=1,default=float))
    print('LR tail',res['D']['LR_tail'],'boot p',res['D']['boot_p']); print('saved',out)
