import json, numpy as np, pandas as pd, importlib.util, sys
spec=importlib.util.spec_from_file_location('tc','theorem_check.py')
src=open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'theorem_check.py')).read().split('out={}')[0]
ns={}; exec(src,ns)
SC,kappa,simulate,loglik,dvec=ns['SC'],ns['kappa'],ns['simulate'],ns['loglik'],ns['dvec']
rng=np.random.default_rng(7); res={}
def acf(x,k): x=x-x.mean(1,keepdims=True); return float((x[:,:-k]*x[:,k:]).mean()/ (x*x).mean())
for name,p in SC.items():
    pi0,pi1,chi,kap=kappa(p); rho=p[0]+p[1]-1
    y=simulate(p,50,40000,rng); d=dvec(y,pi0,p)
    # path-level ACF estimates -> mean and MC s.e. across 50 paths
    a1=[]; a2=[]; s1=[]
    for i in range(50):
        di=d[i:i+1]; a1.append(acf(di,1)); a2.append(acf(di,2)); s1.append(acf(y[i:i+1]**2,1))
    res[name]=dict(kappa=kap,rho=rho,acf1_d=np.mean(a1),acf1_d_se=np.std(a1)/np.sqrt(50),acf1_theory=kap*rho,
                   acf2_d=np.mean(a2),acf2_d_se=np.std(a2)/np.sqrt(50),acf2_theory=kap*rho**2,acf1_y2=np.mean(s1),acf1_y2_se=np.std(s1)/np.sqrt(50))
    print(name,{k:round(v,5) for k,v in res[name].items()},flush=True)
# population one-step gain with s.e. for the weak designs
for name in ['tail_only','weak_tail','weak_persistence']:
    p=SC[name]; pi0,pi1,chi,kap=kappa(p); rho=p[0]+p[1]-1
    g=[]
    for rep in range(8):
        yy=simulate(p,100,5000,rng); g.extend(((loglik(yy,pi0,rho,p)-loglik(yy,pi0,0.0,p))/5000).tolist())
    g=np.array(g); res['gain_'+name]=dict(gain=g.mean(),se=g.std()/np.sqrt(len(g)),pred=0.5*kap**2*rho**2/(1-rho**2))
    print('gain',name,res['gain_'+name],flush=True)
# score variance at rho=0 (part a) with bootstrap s.e.
p=SC['iid_mixture']; pi0,pi1,chi,kap=kappa(p); T=200
s=[]
for rep in range(5):
    y=simulate(p,20000,T,rng); d=dvec(y,pi0,p); s.extend((pi0*pi1*(d[:,:-1]*d[:,1:]).sum(1)).tolist())
s=np.array(s); bs=[np.var(rng.choice(s,len(s))) for _ in range(300)]
res['score_var']=dict(n=len(s),var=float(s.var()),se=float(np.std(bs)),theory=(T-1)*kap**2)
print('score var',res['score_var'])
json.dump(res,open('theorem_check2.json','w'),indent=1,default=float)
