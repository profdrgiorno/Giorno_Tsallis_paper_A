"""Numerical checks of the persistence-information theorem for the centred two-state Student-t HMM.
kappa = pi0*pi1*int (f0-f1)^2/fbar.  (a) exact: at rho=0 the rho-score equals pi0*pi1*sum d_t d_{t+1}
and its variance is (T-1)kappa^2.  (b) small contrast: I_rho/T ~ kappa^2 (1+rho^2)/(1-rho^2)^3 and
one-step log-score gain over the i.i.d. mixture ~ 0.5*kappa^2*rho^2/(1-rho^2)."""
import json, numpy as np, pandas as pd
from scipy.stats import t as tdist
from scipy.integrate import quad
import os; R=os.environ.get('GT_REPO',os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'known_generator'))+'/'
SC={'calibration_anchor':(.96,.94,12.,4.5,.5,1.8),'full_hmm6':(.97,.86,18.,5.2,.7,1.45),
    'scale_only':(.965,.82,8.,8.,.65,1.65)}
st=lambda nu: np.sqrt((nu-2)/nu)
SC.update({'tail_only':(.975,.85,24.,4.6,st(24.),st(4.6)),'weak_tail':(.975,.85,11.,8.,st(11.),st(8.)),
           'weak_persistence':(.72,.47,24.,4.6,st(24.),st(4.6)),'iid_mixture':(.82,.18,24.,4.6,st(24.),st(4.6))})
f=lambda x,nu,s: tdist.pdf(x/s,nu)/s
def kappa(p):
    p00,p11,n0,n1,s0,s1=p; pi0=(1-p11)/(2-p00-p11); pi1=1-pi0
    fb=lambda x: pi0*f(x,n0,s0)+pi1*f(x,n1,s1)
    chi=quad(lambda x:(f(x,n0,s0)-f(x,n1,s1))**2/fb(x),-np.inf,np.inf,limit=400,epsabs=1e-13)[0]
    return pi0,pi1,chi,pi0*pi1*chi
def simulate(p,N,T,rng):
    p00,p11,n0,n1,s0,s1=p; pi0=(1-p11)/(2-p00-p11)
    k=np.zeros((N,T),int); k[:,0]=rng.random(N)>pi0
    u=rng.random((N,T))
    for t in range(1,T): k[:,t]=np.where(k[:,t-1]==0,u[:,t]>p00,u[:,t]>1-p11)
    return np.where(k==0,rng.standard_t(n0,(N,T))*s0,rng.standard_t(n1,(N,T))*s1)
def loglik(y,pi0,rho,p):   # vectorised scaled forward, (pi0,rho) parameterisation, stationary start
    _,_,n0,n1,s0,s1=p; pi1=1-pi0
    P=np.array([[pi0+rho*pi1,pi1-rho*pi1],[pi0-rho*pi0,pi1+rho*pi0]])
    E0=f(y,n0,s0); E1=f(y,n1,s1)
    a0=pi0*E0[:,0]; a1=pi1*E1[:,0]; c=a0+a1; ll=np.log(c); a0/=c; a1/=c
    for t in range(1,y.shape[1]):
        b0=(a0*P[0,0]+a1*P[1,0])*E0[:,t]; b1=(a0*P[0,1]+a1*P[1,1])*E1[:,t]
        c=b0+b1; ll+=np.log(c); a0=b0/c; a1=b1/c
    return ll
def dvec(y,pi0,p):
    _,_,n0,n1,s0,s1=p; f0=f(y,n0,s0); f1=f(y,n1,s1); return (f0-f1)/(pi0*f0+(1-pi0)*f1)
out={}; rng=np.random.default_rng(20261031)
# ---- (a) exact identity at rho=0, i.i.d.-mixture design
p=SC['iid_mixture']; pi0,pi1,chi,kap=kappa(p); T=200; N=20000
y=simulate(p,N,T,rng); h=1e-6
s_fd=(loglik(y,pi0,h,p)-loglik(y,pi0,-h,p))/(2*h)
d=dvec(y,pi0,p); s_th=pi0*pi1*(d[:,:-1]*d[:,1:]).sum(1)
out['a']=dict(T=T,N=N,kappa=kap,max_abs_diff=float(np.max(np.abs(s_fd-s_th))),corr=float(np.corrcoef(s_fd,s_th)[0,1]),
              var_score=float(s_fd.var()),var_theory=(T-1)*kap**2,mean_d=float(d.mean()),var_d=float(d.var()),chi=chi)
print('(a)',out['a'],flush=True)
# ---- per-design kappa, predictions vs archive
rec=pd.read_csv(R+'HMM6_PARAMETER_RECOVERY_ALL.csv'); pm=pd.read_csv(R+'PATH_MATCHED_HMM6_VS_19_MODELS.csv')
rows=[]; Ttr=4800
for name,p in SC.items():
    pi0,pi1,chi,kap=kappa(p); rho=p[0]+p[1]-1
    info=kap**2*(1+rho**2)/(1-rho**2)**3
    sd_pred=1/np.sqrt(Ttr*info) if info>0 else np.inf
    gain_pred=0.5*kap**2*rho**2/(1-rho**2)
    # exact population one-step gain of the true HMM over the true i.i.d. mixture (no estimation), long simulation
    yy=simulate(p,40,5000,rng)
    ll_h=loglik(yy,pi0,rho,p); ll_m=loglik(yy,pi0,0.0,p)
    gain_exact=float((ll_h-ll_m).mean()/5000)
    r=rec[rec.scenario==name].estimated_rho
    g=pm[(pm.scenario==name)&(pm.model=='IIDmix-t')].iloc[0]
    rows.append(dict(design=name,rho=rho,pi0=pi0,chi=chi,kappa=kap,n_eff=Ttr*kap**2,sd_pred=sd_pred,sd_obs=r.std(),
                     gain_pred=gain_pred,gain_exact_pop=gain_exact,gain_obs=g.mean_hmm_minus_competitor,gain_obs_lo=g.bootstrap_ci95_lo,gain_obs_hi=g.bootstrap_ci95_hi))
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv('kappa_by_design.csv',index=False)
# ---- (b) Fisher information at true rho, small vs large contrast (score variance by simulation)
for name in ['tail_only','weak_tail','weak_persistence','full_hmm6']:
    p=SC[name]; pi0,pi1,chi,kap=kappa(p); rho=p[0]+p[1]-1; T=2000; N=600
    yy=simulate(p,N,T,rng); h=1e-5
    s=(loglik(yy,pi0,rho+h,p)-loglik(yy,pi0,rho-h,p))/(2*h)
    out['b_'+name]=dict(T=T,N=N,info_per_obs_sim=float(s.var()/T),se=float(s.var()/T*np.sqrt(2/(N-1))),
                         info_per_obs_theory=kap**2*(1+rho**2)/(1-rho**2)**3)
    print(name,out['b_'+name],flush=True)
json.dump(out,open('theorem_check.json','w'),indent=1)
