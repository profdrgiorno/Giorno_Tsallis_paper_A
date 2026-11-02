"""Harness: run the repository's own EGARCH competitor code without numba (not installable here).
numba is stubbed with identity decorators; the sequential EGARCH filter is replaced by an
equivalent C routine (ctypes). clip=0 reproduces the original sentinel behaviour exactly."""
import sys, types, ctypes, subprocess, importlib, os
REPO = os.environ.get('GT_REPO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'known_generator'))
HERE = os.path.dirname(os.path.abspath(__file__))
nb = types.ModuleType('numba')
def njit(*a, **k):
    if a and callable(a[0]): return a[0]
    return lambda f: f
nb.njit = njit; sys.modules['numba'] = nb
sys.path.insert(0, REPO)
C = r'''
#include <math.h>
int egarch(const double*y,int n,const double*p,int np,double initvar,double logconst,double ehabs,int clip,
           double*scores,double*hs,double*errs){
  double mu=p[0],omega=p[1],alpha=p[2],beta=p[3],gamma=p[4],nu=p[np-1];
  double hprev=initvar,eprev=0.0,loghprev=log(initvar),h; int flag=0;
  for(int t=0;t<n;t++){
    if(t==0){h=initvar;}
    else{
      double zprev=eprev/sqrt(hprev);
      double lh=omega+beta*loghprev+alpha*(fabs(zprev)-ehabs)+gamma*zprev;
      if(lh<-24||lh>24){
        if(clip){flag=1; lh=lh<-24?-24:24;}
        else{for(int v=t;v<n;v++){scores[v]=-1e8;hs[v]=1e-20;errs[v]=0.0;} return 2;}
      }
      h=exp(lh);
    }
    if(!isfinite(h)||h<1e-14||h>1e14){
      for(int v=t;v<n;v++){scores[v]=-1e8;hs[v]=1e-20;errs[v]=0.0;} return 3;}
    double err=y[t]-mu;
    scores[t]=logconst-0.5*log(h)-(nu+1.0)*0.5*log1p(err*err/((nu-2.0)*h));
    hs[t]=h; errs[t]=err; eprev=err; hprev=h; loghprev=log(h);
  }
  return flag;
}
'''
src = os.path.join(HERE, 'egarch.c'); so = os.path.join(HERE, 'egarch.so')
open(src, 'w').write(C)
subprocess.check_call(['gcc', '-O2', '-shared', '-fPIC', '-o', so, src, '-lm'])
lib = ctypes.CDLL(so)
import numpy as np
dp = ctypes.POINTER(ctypes.c_double)
lib.egarch.argtypes = [dp, ctypes.c_int, dp, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, dp, dp, dp]
CLIP = {'on': 0}
LAST = {'flag': 0}
comp = importlib.import_module('02_blind_econometric_competitors')
_orig = comp._garch_filter
def fast_filter(y, pars, typ, initial_var, logconst, ehabs):
    if typ != 3: return _orig(y, pars, typ, initial_var, logconst, ehabs)
    y = np.ascontiguousarray(y, dtype=np.float64); pars = np.ascontiguousarray(pars, dtype=np.float64); n = len(y)
    s = np.empty(n); h = np.empty(n); e = np.empty(n)
    f = lambda a: a.ctypes.data_as(dp)
    LAST['flag'] = lib.egarch(f(y), n, f(pars), len(pars), float(initial_var), float(logconst), float(ehabs), CLIP['on'], f(s), f(h), f(e))
    return s, h, e
comp._garch_filter = fast_filter
import giorno_models as gm
