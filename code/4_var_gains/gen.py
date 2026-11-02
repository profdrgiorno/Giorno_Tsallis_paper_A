"""Exact regeneration of the archived 4,000 HMM-6 paths (returns only) from DGP_MANIFEST seeds,
replicating giorno_models.simulate_hmm6 without its (unneeded) forward filter."""
import numpy as np, pandas as pd
st=lambda nu: np.sqrt((nu-2)/nu)
SC={'calibration_anchor':(.96,.94,12.,4.5,.5,1.8),'full_hmm6':(.97,.86,18.,5.2,.7,1.45),'scale_only':(.965,.82,8.,8.,.65,1.65),
    'tail_only':(.975,.85,24.,4.6,st(24.),st(4.6)),'weak_tail':(.975,.85,11.,8.,st(11.),st(8.)),
    'weak_persistence':(.72,.47,24.,4.6,st(24.),st(4.6)),'iid_mixture':(.82,.18,24.,4.6,st(24.),st(4.6)),
    'single_student_t':(.975,.85,7.5,7.5,st(7.5),st(7.5))}
def regen(scenario,seed,n=6000,burn=600):
    p00,p11,nu0,nu1,s0,s1=SC[scenario]; rng=np.random.default_rng(int(seed)); total=n+burn
    u=rng.random(total); st_=np.empty(total,np.int8); pi0=(1-p11)/(2-p00-p11)
    st_[0]=0 if u[0]<pi0 else 1
    for t in range(1,total): st_[t]=0 if (u[t]<(p00 if st_[t-1]==0 else 1-p11)) else 1
    z=rng.standard_t(np.where(st_==0,nu0,nu1))*np.where(st_==0,s0,s1)
    y=z[burn:]; return np.array([float('%.12g'%v) for v in y]), st_[burn:]
