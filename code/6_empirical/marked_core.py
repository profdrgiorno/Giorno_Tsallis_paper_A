"""Daily OHLC version of the marked-HMM VaR exercise.
Returns alternate: overnight 100*log(O_t/C_{t-1}) (mark=1) and intraday 100*log(C_t/O_t) (mark=0).
M1: centred Student-t (nu, s, overnight multiplier m): 3 parameters.
M2: two-state centred Student-t HMM (p00,p11,nu0,nu1,s0,s1) with state-specific overnight multipliers (m0,m1): 8 parameters.
M2r: M2 with nu0=nu1 (tail-equality null): 7 parameters."""
import ctypes, json, sys, numpy as np, pandas as pd
from scipy.optimize import minimize, brentq
from scipy.special import expit, logit, gammaln
from scipy.stats import t as tdist, chi2, norm
import os
lib=ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),'mhmm.so' if os.name!='nt' else 'mhmm.dll'))
dp=ctypes.POINTER(ctypes.c_double); ip=ctypes.POINTER(ctypes.c_int)
lib.mhmm_ll.restype=ctypes.c_double; lib.mhmm_ll.argtypes=[dp,ip,ctypes.c_int,dp,dp]
def ll_hmm(y,o,p,pred=None):
    y=np.ascontiguousarray(y,float); o=np.ascontiguousarray(o,np.int32); p=np.ascontiguousarray(p,float)
    return lib.mhmm_ll(y.ctypes.data_as(dp),o.ctypes.data_as(ip),len(y),p.ctypes.data_as(dp),pred.ctypes.data_as(dp) if pred is not None else None)
NUMAX=100.
def tr_hmm(q,restricted=False):
    p00,p11=expit(q[0]),expit(q[1]); nu0=2.05+(NUMAX-2.05)*expit(q[2]); nu1=nu0 if restricted else 2.05+(NUMAX-2.05)*expit(q[3])
    k=3 if restricted else 4
    s0,s1,m0,m1=np.exp(q[k:k+4]); return np.array([p00,p11,nu0,nu1,s0,s1,m0,m1])
def fit_hmm(y,o,restricted=False,starts=None):
    sd=np.std(y[o==0]); so=np.std(y[o==1])/sd
    if starts is None:
        starts=[]
        for (p00,p11,n0,n1,f0,f1) in [(.97,.95,4,12,.7,1.6),(.9,.9,3,8,.8,1.3),(.98,.9,6,30,.6,1.8),(.6,.6,3,20,.9,1.1),(.2,.8,2.6,30,1.,1.)]:
            q=[logit(p00),logit(p11),logit((n0-2.05)/(NUMAX-2.05))]+([] if restricted else [logit((n1-2.05)/(NUMAX-2.05))])+list(np.log([f0*sd,f1*sd,so,so]))
            starts.append(np.array(q))
    best=None
    for q0 in starts:
        f=lambda q:-ll_hmm(y,o,tr_hmm(q,restricted))
        r=minimize(f,q0,method='L-BFGS-B',options={'maxiter':800})
        if best is None or r.fun<best.fun: best=r
    return tr_hmm(best.x,restricted),-best.fun,best
def fit_t(y,o):
    def f(q):
        nu=2.05+(NUMAX-2.05)*expit(q[0]); s=np.exp(q[1])*np.where(o==1,np.exp(q[2]),1.0)
        return -np.sum(tdist.logpdf(y/s,nu)-np.log(s))
    best=None
    for n0 in (3.,5.,10.):
        r=minimize(f,np.array([logit((n0-2.05)/(NUMAX-2.05)),np.log(np.std(y)),0.]),method='L-BFGS-B')
        if best is None or r.fun<best.fun: best=r
    q=best.x; return np.array([2.05+(NUMAX-2.05)*expit(q[0]),np.exp(q[1]),np.exp(q[2])]),-best.fun
def mix_quantile(a,w0,nu0,s0,nu1,s1):
    F=lambda x:w0*tdist.cdf(x/s0,nu0)+(1-w0)*tdist.cdf(x/s1,nu1)-a
    lo=min(tdist.ppf(a,nu0)*s0,tdist.ppf(a,nu1)*s1); hi=max(tdist.ppf(a,nu0)*s0,tdist.ppf(a,nu1)*s1)
    return lo if hi-lo<1e-12 else brentq(F,lo-1e-9,hi+1e-9)
def nw_se(x,L):
    x=x-x.mean(); n=len(x); g=x@x/n
    for l in range(1,L+1): g+=2*(1-l/(L+1))*(x[l:]@x[:-l])/n
    return np.sqrt(g/n)
def kupiec(x,n,a):
    pi=x/n; ll1=(n-x)*np.log(1-pi)+(x*np.log(pi) if x>0 else 0); lr=-2*((n-x)*np.log(1-a)+x*np.log(a)-ll1); return float(chi2.sf(lr,1))
