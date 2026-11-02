"""Identification collapse: dispersion of rho-hat versus n_eff = T*kappa^2 (Theorem 2).
Two-state centred Student-t HMM with rho=0.825 (p00=.975, p11=.85), fitted without access to the truth.
Families: (S) scale contrast, nu0=nu1=8, s0=1, s1=r; (U) tail contrast at equal unit variance, nu1=4.6, nu0 varied."""
import ctypes, json, sys, os, numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import t as tdist
from scipy.integrate import quad
HERE=os.path.dirname(os.path.abspath(__file__))
L=ctypes.CDLL(os.path.join(HERE,'hmmfast.so')); dp=ctypes.POINTER(ctypes.c_double)
L.hmm_ll.restype=ctypes.c_double; L.hmm_ll.argtypes=[dp,ctypes.c_int,dp]
def ll(b,y): y=np.ascontiguousarray(y); b=np.ascontiguousarray(b,float); return L.hmm_ll(y.ctypes.data_as(dp),len(y),b.ctypes.data_as(dp))
tr=lambda q: np.array([expit(q[0]),expit(q[1]),2.05+97.95*expit(q[2]),2.05+97.95*expit(q[3]),np.exp(q[4]),np.exp(q[5])])
inv=lambda b: np.array([logit(b[0]),logit(b[1]),logit((b[2]-2.05)/97.95),logit((b[3]-2.05)/97.95),np.log(b[4]),np.log(b[5])])
def fit(y):
    sd=y.std(); best=None
    for st in ([.9,.9,10.,4.,.8*sd,1.2*sd],[.7,.7,6.,6.,.7*sd,1.4*sd],[.97,.8,20.,5.,.9*sd,1.1*sd]):
        r=minimize(lambda q:-ll(tr(q),y)/len(y),inv(np.array(st)),method='L-BFGS-B',options={'maxiter':400})
        if best is None or r.fun<best.fun: best=r
    b=tr(best.x); return b[0]+b[1]-1, b
def kappa(p):
    p00,p11,n0,n1,s0,s1=p; pi0=(1-p11)/(2-p00-p11); pi1=1-pi0
    f0=lambda x: tdist.pdf(x/s0,n0)/s0; f1=lambda x: tdist.pdf(x/s1,n1)/s1
    return pi0*pi1*quad(lambda x:(f0(x)-f1(x))**2/(pi0*f0(x)+pi1*f1(x)),-np.inf,np.inf,limit=400)[0]
def sim(p,T,rng):
    p00,p11,n0,n1,s0,s1=p; pi0=(1-p11)/(2-p00-p11); k=np.empty(T,int); k[0]=rng.random()>pi0; u=rng.random(T)
    for t in range(1,T): k[t]=(u[t]>p00) if k[t-1]==0 else (u[t]>1-p11)
    return np.where(k==0,rng.standard_t(n0,T)*s0,rng.standard_t(n1,T)*s1)
us=lambda nu: np.sqrt((nu-2)/nu)
DES=[('S',r,(.975,.85,8.,8.,1.,r)) for r in (1.1,1.25,1.5,2.0,3.0)]+[('U',n0,(.975,.85,n0,4.6,us(n0),us(4.6))) for n0 in (6.,12.,24.,60.)]
if __name__=='__main__':
    part,nparts,R=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]); TS=(1200,4800)
    cells=[(fam,v,p,T) for fam,v,p in DES for T in TS]
    out=[]; rng0=np.random.SeedSequence(20261031)
    for ci,(fam,v,p,T) in enumerate(cells):
        if ci%nparts!=part: continue
        kap=kappa(p); rng=np.random.default_rng(np.random.SeedSequence([20261031,ci]))
        rh=[fit(sim(p,T,rng))[0] for _ in range(R)]
        out.append(dict(family=fam,value=v,T=T,kappa=kap,neff=T*kap**2,rho_hat=rh))
        print(fam,v,T,round(kap,5),round(T*kap**2,2),'sd',round(np.std(rh),3),flush=True)
        json.dump(out,open(f'collapse_p{part}.json','w'))
