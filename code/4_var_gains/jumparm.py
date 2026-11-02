"""Jump arm of the known-generator study. Same archived paths (exact regeneration from manifest seeds);
jumps added with a separate seeded stream: Y = X - B*theta*Z, B~Bern(eps), Z~Gamma(2,1), common random numbers.
Models fitted on the first 4,800 observations: Student-t (location, scale, nu), GARCH(1,1)-t, APARCH(1,1)-t, HMM-6.
Holdout (1,200 obs): one-step VaR at 95/99/99.5% from each model and the TRUE conditional VaR of the DGP."""
import ctypes, json, sys, time, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, roots_genlaguerre, gammaln
from scipy.stats import t as tdist, chi2
import os; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); from gen import regen, SC
# build first: cc -O2 -shared -fPIC -o mhmm.so mhmm.c ; cc -O2 -shared -fPIC -o aparch2.so aparch2.c -lm
dp=ctypes.POINTER(ctypes.c_double); ip=ctypes.POINTER(ctypes.c_int)
H6=ctypes.CDLL(os.path.join(HERE,'mhmm.so')); H6.mhmm_ll.restype=ctypes.c_double; H6.mhmm_ll.argtypes=[dp,ip,ctypes.c_int,dp,dp]
AP=ctypes.CDLL(os.path.join(HERE,'aparch2.so')); AP.aparch_h.restype=ctypes.c_double; AP.aparch_h.argtypes=[dp,ctypes.c_int,dp,ctypes.c_double,dp]
LEV=(0.05,0.01,0.005); K=2.0; XN,WN=roots_genlaguerre(32,K-1); WN=WN/np.exp(gammaln(K))
c_=lambda a: np.ascontiguousarray(a,dtype=float)
# ---------- models
def hmm_ll(b,y,pred=None):
    yy=c_(y); o=np.zeros(len(y),np.int32); bb=c_(np.r_[b,1.,1.])
    return H6.mhmm_ll(yy.ctypes.data_as(dp),o.ctypes.data_as(ip),len(yy),bb.ctypes.data_as(dp),pred.ctypes.data_as(dp) if pred is not None else None)
def hmm_tr(q): return np.array([expit(q[0]),expit(q[1]),2.05+97.95*expit(q[2]),2.05+97.95*expit(q[3]),np.exp(q[4]),np.exp(q[5])])
def hmm_inv(b): return np.array([logit(b[0]),logit(b[1]),logit((b[2]-2.05)/97.95),logit((b[3]-2.05)/97.95),np.log(b[4]),np.log(b[5])])
def ap_ll(b,y,h0,hout=None):
    return AP.aparch_h(c_(y).ctypes.data_as(dp),len(y),c_(b).ctypes.data_as(dp),float(h0),hout.ctypes.data_as(dp) if hout is not None else None)
def ap_full(q,kind):
    if kind=='garch': return np.array([q[0],np.exp(q[1]),expit(q[2])*.999,0.,expit(q[3])*.999,2.,2.05+97.95*expit(q[4])])
    return np.array([q[0],np.exp(q[1]),expit(q[2])*.999,np.tanh(q[3]),expit(q[4])*.999,.3+3.7*expit(q[5]),2.05+97.95*expit(q[6])])
def ap_inv(b,kind):
    if kind=='garch': return np.array([b[0],np.log(b[1]),logit(b[2]/.999),logit(b[4]/.999),logit((b[6]-2.05)/97.95)])
    return np.array([b[0],np.log(b[1]),logit(b[2]/.999),np.arctanh(b[3]),logit(b[4]/.999),logit((b[5]-.3)/3.7),logit((b[6]-2.05)/97.95)])
def fit(f,starts):
    best=None
    for q0 in starts:
        try: r=minimize(f,q0,method='L-BFGS-B',options={'maxiter':500})
        except Exception: continue
        if np.isfinite(r.fun) and (best is None or r.fun<best.fun): best=r
    return best
def fit_t(y,warm=None):
    f=lambda q: -np.mean(tdist.logpdf((y-q[0])/np.exp(q[1]),2.05+97.95*expit(q[2]))-q[1])
    st=[np.array([np.median(y),np.log(.7*y.std()),logit((5-2.05)/97.95)])]+([warm] if warm is not None else [])
    r=fit(f,st); q=r.x; return dict(q=q,b=np.array([q[0],np.exp(q[1]),2.05+97.95*expit(q[2])]),ll=-r.fun*len(y))
def fit_ap(y,kind,warm=None):
    h0=float(y.var())
    def f(q):
        b=ap_full(q,kind)
        if b[2]+b[4]>=0.999: return 1e6
        v=ap_ll(b,y,h0); return -v/len(y) if np.isfinite(v) and v>-1e299 else 1e6
    v=y.var()
    st=[ap_inv(np.array([0.,.1*v,.08,0.,.88,2. if kind=='garch' else 1.6,6.]),kind),ap_inv(np.array([0.,.25*v,.15,0.,.65,2. if kind=='garch' else 1.8,4.5]),kind)]+([warm] if warm is not None else [])
    r=fit(f,st); return dict(q=r.x,b=ap_full(r.x,kind),ll=-r.fun*len(y),h0=h0)
def fit_hmm(y,p,warm=None):
    f=lambda q: -hmm_ll(hmm_tr(q),y)/len(y)
    st=[hmm_inv(np.array(p)),hmm_inv(np.array([.9,.9,10.,4.,.8*y.std(),1.3*y.std()]))]+([warm] if warm is not None else [])
    r=fit(f,st); return dict(q=r.x,b=hmm_tr(r.x),ll=-r.fun*len(y))
# ---------- quantiles of mixtures by vectorised bisection
def bisect(cdf,a,lo,hi,it=60):
    lo=np.full_like(a,lo,dtype=float); hi=np.full_like(a,hi,dtype=float)
    for _ in range(it):
        m=0.5*(lo+hi); v=cdf(m)<a; lo=np.where(v,m,lo); hi=np.where(v,hi,m)
    return 0.5*(lo+hi)
def true_var(y,p,eps,theta,n_tr):
    """true one-step predictive quantiles given past observed y (filter uses jump-convolved emissions)"""
    p00,p11,n0,n1,s0,s1=p; P=np.array([[p00,1-p00],[1-p11,p11]]); pi=np.array([1-p11,1-p00])/(2-p00-p11)
    f=lambda x,nu,s: tdist.pdf(x/s,nu)/s
    def g(x,nu,s):
        if eps==0: return f(x,nu,s)
        return (1-eps)*f(x,nu,s)+eps*sum(w*f(x+theta*z,nu,s) for z,w in zip(XN,WN))
    E=np.stack([g(y,n0,s0),g(y,n1,s1)],1); a=pi.copy(); W=np.empty((len(y),2))
    for t in range(len(y)):
        W[t]=a; post=a*E[t]; post/=post.sum(); a=post@P
    w=W[n_tr:,0]
    def Fj(x,nu,s):
        base=tdist.cdf(x/s,nu)
        if eps==0: return base
        return (1-eps)*base+eps*sum(wq*tdist.cdf((x+theta*z)/s,nu) for z,wq in zip(XN,WN))
    out={}
    for a_ in LEV:
        cdf=lambda x: w*Fj(x,n0,s0)+(1-w)*Fj(x,n1,s1)
        out[a_]=bisect(cdf,np.full(len(w),a_),-60.,5.)
    return out
def kupiec(x,n,a):
    pi=x/n; ll1=(n-x)*np.log(1-pi)+(x*np.log(pi) if x>0 else 0); return float(chi2.sf(-2*((n-x)*np.log(1-a)+x*np.log(a)-ll1),1))
def basel_k(x):
    return 3.0 if x<=4 else {5:3.4,6:3.5,7:3.65,8:3.75,9:3.85}.get(x,4.0)
def forecasts(y,n_tr,mods,p_true):
    out={}
    b=mods['t']['b']; q={a:np.full(len(y)-n_tr,b[0]+b[1]*tdist.ppf(a,b[2])) for a in LEV}; out['Student-t']=q
    for nm,key in (('GARCH-t','garch'),('APARCH-t','aparch')):
        m=mods[key]; h=np.empty(len(y)); ap_ll(m['b'],y,m['h0'],h); sd=np.sqrt(h[n_tr:]); nu=m['b'][6]
        out[nm]={a:m['b'][0]+sd*np.sqrt((nu-2)/nu)*tdist.ppf(a,nu) for a in LEV}
    m=mods['hmm']; b=m['b']; pred=np.empty(len(y)); hmm_ll(b,y,pred); w=pred[n_tr:]
    out['HMM-6']={a:bisect(lambda x: w*tdist.cdf(x/b[4],b[2])+(1-w)*tdist.cdf(x/b[5],b[3]),np.full(len(w),a),-60.,5.) for a in LEV}
    return out
def evaluate(y,n_tr,fc,tv):
    yt=y[n_tr:]; rows=[]
    for nm,q in fc.items():
        for a in LEV:
            v=-q[a]; vt=-tv[a]; br=yt<q[a]; x=int(br.sum()); x250=int(br[-250:].sum())
            rows.append(dict(model=nm,level=a,breach_rate=x/len(yt),kupiec_p=kupiec(x,len(yt),a),mean_var=float(v.mean()),mean_true_var=float(vt.mean()),
                             shortfall=float((vt-v).mean()),rel_shortfall=float(((vt-v)/vt).mean()),pinball=float(((a-br)*(yt-q[a])).mean()),
                             pinball_true=float(((a-(yt<tv[a]))*(yt-tv[a])).mean()),exc_last250=x250,basel_k=basel_k(x250) if a==0.01 else None))
    return rows
if __name__=='__main__':
    eps,theta=float(sys.argv[1]),float(sys.argv[2]); nper=int(sys.argv[3]); part,nparts=int(sys.argv[4]),int(sys.argv[5]); outfn=sys.argv[6]
    man=pd.read_csv(os.environ.get('DGP_MANIFEST',os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'known_generator'),'DGP_MANIFEST.csv')))
    man=man[man.rep<nper].reset_index(drop=True)
    recs=[]; t00=time.time()
    for i,e in man.iterrows():
        if i%nparts!=part: continue
        p=SC[e.scenario]; x,_=regen(e.scenario,e.seed); n_tr=int(e.n_train)
        rj=np.random.default_rng([int(e.seed),20261031]); B=rj.random(len(x))<eps; Z=rj.gamma(K,1.0,len(x))
        yj=x-B*Z*theta
        rec=dict(id=e.id,scenario=e.scenario,rep=int(e.rep),n_jumps_train=int(B[:n_tr].sum()),n_jumps_test=int(B[n_tr:].sum()))
        for arm,y,ep in (('nojump',x,0.0),('jump',yj,eps)):
            tr=y[:n_tr]; warm=rec.get('warm')
            mods={'t':fit_t(tr,None if warm is None else warm['t']),'garch':fit_ap(tr,'garch',None if warm is None else warm['garch']),
                  'aparch':fit_ap(tr,'aparch',None if warm is None else warm['aparch']),'hmm':fit_hmm(tr,p,None if warm is None else warm['hmm'])}
            if arm=='nojump': rec['warm']={k:v['q'] for k,v in mods.items()}
            for k,v in mods.items():
                rec[f'{arm}_{k}_b']=v['b'].tolist(); rec[f'{arm}_{k}_ll']=v['ll']
            tv=true_var(y,p,ep,theta,n_tr); fc=forecasts(y,n_tr,mods,p)
            rec[f'{arm}_eval']=evaluate(y,n_tr,fc,tv)
        rec.pop('warm'); recs.append(rec)
        if len(recs)%10==0:
            json.dump(recs,open(outfn,'w')); print(len(recs),'paths',round(time.time()-t00),'s',flush=True)
    json.dump(recs,open(outfn,'w')); print('done',len(recs),round(time.time()-t00),flush=True)
