import glob, numpy as np, pandas as pd
rows=[]
for f in sorted(glob.glob('bt_*.csv')):
    d=pd.read_csv(f); 
    if len(d)<50: continue
    q=np.quantile(np.abs(d.Zb),.95); qn=np.quantile(d.neffb,.95)
    rows.append(dict(dgp=d.dgp[0],n=len(d),rej_boot=float((np.abs(d.Z)>q).mean()),rej_asym=float((np.abs(d.Z)>1.96).mean()),
        crit_boot=q,mean_Z=d.Z.mean(),sd_Z=d.Z.std(),sd_Zb=d.Zb.std(),neff_med=d.neff.median(),neffb_q95=qn))
R=pd.DataFrame(rows); R.to_csv('boot_summary.csv',index=False); print(R.round(3).to_string())
