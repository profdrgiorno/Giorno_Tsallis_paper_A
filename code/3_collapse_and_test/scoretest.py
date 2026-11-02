"""Size and power of the persistence score test (Theorem 2 iii) with an estimated i.i.d. Student-t mixture:
Z = sqrt(T-1) * lag-one autocorrelation of d-hat(Y_t),  two-sided 5%."""
import json, sys, numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, gammaln
from scipy.stats import norm, t as tdist
from scipy.integrate import quad
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from collapse import sim, kappa
def lt(y,nu,s): return gammaln((nu+1)/2)-gammaln(nu/2)-.5*np.log(np.pi*nu)-np.log(s)-.5*(nu+1)*np.log1p((y/s)**2/nu)
def fit_mix(y):
    def tr(q): return expit(q[0]),2.05+97.95*expit(q[1]),2.05+97.95*expit(q[2]),np.exp(q[3]),np.exp(q[4])
    def f(q):
        w,n0,n1,s0,s1=tr(q); return -np.mean(np.logaddexp(np.log(w)+lt(y,n0,s0),np.log1p(-w)+lt(y,n1,s1)))
    sd=y.std(); best=None
    for st in ([.5,8,8,.7*sd,1.4*sd],[.7,20,5,.8*sd,1.2*sd]):
        q0=[logit(st[0]),logit((st[1]-2.05)/97.95),logit((st[2]-2.05)/97.95),np.log(st[3]),np.log(st[4])]
        r=minimize(f,q0,method='L-BFGS-B',options={'maxiter':500})
        if best is None or r.fun<best.fun: best=r
    return tr(best.x)
def ztest(y):
    w,n0,n1,s0,s1=fit_mix(y); f0=np.exp(lt(y,n0,s0)); f1=np.exp(lt(y,n1,s1)); d=(f0-f1)/(w*f0+(1-w)*f1)
    dc=d-d.mean(); ac=(dc[:-1]*dc[1:]).mean()/dc.var(); return np.sqrt(len(y)-1)*ac
if __name__=='__main__':
    part,nparts,R=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]); T=4800; out=[]; ci=0
    for lab,base in (('scale 1.5',(8.,8.,1.,1.5)),('scale 2.0',(8.,8.,1.,2.0)),('tail-only',(24.,4.6,np.sqrt(22/24),np.sqrt(2.6/4.6)))):
        for rho in (0.0,0.1,0.2,0.4):
            ci+=1
            if ci%nparts!=part: continue
            pi0=0.7; p00=pi0+rho*(1-pi0); p11=(1-pi0)+rho*pi0; p=(p00,p11)+base
            kap=kappa(p); rng=np.random.default_rng([2026,ci])
            Z=np.array([ztest(sim(p,T,rng)) for _ in range(R)])
            rej=float((np.abs(Z)>1.96).mean()); mu=np.sqrt(T-1)*kap*rho
            out.append(dict(design=lab,rho=rho,kappa=kap,reject=rej,theory=float(norm.sf(1.96-mu)+norm.cdf(-1.96-mu)),meanZ=float(Z.mean()),R=R))
            print(out[-1],flush=True); json.dump(out,open(f'score_p{part}.json','w'))
