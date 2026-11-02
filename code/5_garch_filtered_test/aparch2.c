#include <math.h>
/* same recursion as aparch.c, also returns the conditional variance path */
double aparch_h(const double*y,int n,const double*p,double h0,double*hout){
  double mu=p[0],om=p[1],al=p[2],ga=p[3],be=p[4],de=p[5],nu=p[6];
  double c=lgamma(0.5*(nu+1))-lgamma(0.5*nu)-0.5*log(M_PI*(nu-2)),h=h0,ep=0,ll=0;
  for(int t=0;t<n;t++){
    if(t>0){double q=om+al*pow(fabs(ep)-ga*ep,de)+be*pow(h,0.5*de); if(q<1e-20)q=1e-20; h=pow(q,2.0/de);}
    if(!isfinite(h)||h>1e10) return -1e300;
    if(hout) hout[t]=h;
    double e=y[t]-mu; ll+=c-0.5*log(h)-0.5*(nu+1)*log1p(e*e/((nu-2)*h)); ep=e;
  }
  return ll;
}
