"""Shared helper: load the repository's GARCH-HMM module (numba if installed, plain Python otherwise) and
refit G2 from the fitted G1 solution so that the nested model can never end below G1."""
import sys, os, types, importlib
import numpy as np
from scipy.optimize import minimize
def load_repo(repo_dir):
    try:
        import numba  # noqa: F401
    except ImportError:
        nb=types.ModuleType('numba')
        def njit(*a,**k):
            if a and callable(a[0]): return a[0]
            return lambda f:f
        nb.njit=njit; sys.modules['numba']=nb
    sys.path.insert(0,os.path.abspath(repo_dir))
    return importlib.import_module('GT_GARCH_HMM_Joint_Experiment')
def nested_refit(G,x,st,m1,m2,maxiter=220):
    w=min(max(m1['p00'],1e-3),1-1e-3)
    init=G.pack(m1['vbar'],m1['alpha'],m1['beta'],m1['nu0'],m1['nu1'],w,1-w)
    empvar=max(float(np.var(x)),1e-8)
    bounds=[(np.log(empvar*0.01),np.log(empvar*100)),(-5,7),(-7,7),(2.05,65.0),(2.05,65.0),(-5.0,8.0),(-5.0,8.0)]
    init=np.clip(init,[b[0]+1e-6 for b in bounds],[b[1]-1e-6 for b in bounds])
    obj=lambda p:-G.loglik_path(x,st,*G.unpack(p,True),True)/len(x)
    r=minimize(obj,init,method='L-BFGS-B',bounds=bounds,options={'maxiter':maxiter,'ftol':1e-10,'gtol':3e-6,'maxls':35,'maxcor':15})
    ll=-r.fun*len(x); m2=dict(m2); m2['starts']=list(m2['starts'])+[{'success':bool(r.success),'loglik':float(ll),'nit':int(r.nit),'message':'nested start at fitted G1'}]
    m2['loglik_original_starts']=m2['loglik_train']
    if ll>m2['loglik_train']+1e-6:
        vbar,alpha,beta,nu0,nu1,p00,p11=G.unpack(r.x,True)
        if nu0>nu1: nu0,nu1,p00,p11=nu1,nu0,p11,p00
        m2.update({'vbar':vbar,'omega':vbar*(1-alpha-beta),'alpha':alpha,'beta':beta,'persistence':alpha+beta,'nu0':nu0,'nu1':nu1,
                   'q0':(nu0+3)/(nu0+1),'q1':(nu1+3)/(nu1+1),'p00':p00,'p11':p11,'loglik_train':float(ll),'mean_loglik_train':float(-r.fun),
                   'aic_train':-2*ll+14,'optimization_success':bool(r.success),'selected_start':'nested G1 start'})
    m2['G2_minus_G1_loglik']=m2['loglik_train']-m1['loglik_train']
    return m2
