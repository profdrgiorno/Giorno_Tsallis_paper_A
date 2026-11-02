#include <math.h>
/* centred two-state Student-t HMM log-likelihood, constants hoisted; p={p00,p11,nu0,nu1,s0,s1} */
double hmm_ll(const double*y,int n,const double*p){
  double p00=p[0],p11=p[1],n0=p[2],n1=p[3],s0=p[4],s1=p[5];
  double c0=lgamma(.5*(n0+1))-lgamma(.5*n0)-.5*log(M_PI*n0)-log(s0), c1=lgamma(.5*(n1+1))-lgamma(.5*n1)-.5*log(M_PI*n1)-log(s1);
  double pi0=(1-p11)/(2-p00-p11),post0=pi0,ll=0;
  for(int t=0;t<n;t++){
    double q0=(t==0)?pi0:post0*p00+(1-post0)*(1-p11);
    double z0=y[t]/s0,z1=y[t]/s1;
    double a=log(q0)+c0-.5*(n0+1)*log1p(z0*z0/n0), b=log1p(-q0)+c1-.5*(n1+1)*log1p(z1*z1/n1), m=a>b?a:b;
    double l=m+log(exp(a-m)+exp(b-m)); post0=exp(a-l); ll+=l;
  }
  return ll;
}
