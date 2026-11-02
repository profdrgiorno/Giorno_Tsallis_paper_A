import pandas as pd, numpy as np
import os; R=os.environ.get('GT_REPO',os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','known_generator'))+'/'
a=pd.concat([pd.read_csv(f'./alpha_bound_all_p{i}.csv') for i in (0,1)]).drop_duplicates('id')
print('fits',len(a),'penalised',int(a.penalized.sum()),'optimizer not successful',int((~a.success).sum()))
order=['calibration_anchor','full_hmm6','scale_only','tail_only','weak_tail','weak_persistence','iid_mixture','single_student_t']
pm=pd.read_csv(R+'PATH_MATCHED_HMM6_VS_19_MODELS.csv'); pm=pm[pm.model=='EGARCH11-t'].set_index('scenario')
rng=np.random.default_rng(7); rows=[]
for s in order:
    x=a[a.scenario==s]; d=x.hmm_minus_egarch.values
    b=[rng.choice(d,len(d)).mean() for _ in range(2000)]; lo,hi=np.percentile(b,[2.5,97.5])
    rows.append(dict(scenario=s,n=len(x),mean_gain_repaired=d.mean(),ci_lo=lo,ci_hi=hi,label=('HMM better' if lo>0 else 'HMM worse' if hi<0 else 'tie'),
        orig_valid_gain=pm.loc[s,'mean_hmm_minus_competitor'],orig_ci_lo=pm.loc[s,'bootstrap_ci95_lo'],orig_ci_hi=pm.loc[s,'bootstrap_ci95_hi'],
        share_alpha_at_bound=(x.alpha<1e-6).mean(),mean_alpha=x.alpha.mean(),pen=int(x.penalized.sum())))
o=pd.DataFrame(rows); pd.set_option('display.width',250); print(o.round(5).to_string(index=False))
o.to_csv('./egarch_repaired_summary.csv',index=False)
a.to_csv('./egarch_repaired_all_paths.csv',index=False)
