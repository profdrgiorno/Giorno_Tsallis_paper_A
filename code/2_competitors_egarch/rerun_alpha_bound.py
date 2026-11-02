import sys; sys.path.insert(0,'.')
import harness as H, numpy as np, pandas as pd, time
gm, comp = H.gm, H.comp
_sb = comp.spec_bounds
def sb(name, y):
    n, x0, b = _sb(name, y)
    if name == 'EGARCH11-t':
        b = list(b); b[2] = (0.0, 1.0); x0 = np.asarray(x0, float).copy()
    return n, x0, b
comp.spec_bounds = sb
REPO=H.REPO
man=pd.read_csv(f'{REPO}/DGP_MANIFEST.csv'); rec=pd.read_csv(f'{REPO}/HMM6_PARAMETER_RECOVERY_ALL.csv').set_index('id')
tab=gm.make_scenario_table(); mindex=comp.ALL.index('EGARCH11-t')
ids = sys.argv[1].split(',') if len(sys.argv)>1 and sys.argv[1]!='ALL' else None
part = int(sys.argv[2]) if len(sys.argv)>2 else 0; nparts = int(sys.argv[3]) if len(sys.argv)>3 else 1
outfn = sys.argv[4] if len(sys.argv)>4 else 'alpha_bound_results.csv'
if ids is None: ids = list(man.id)
rows=[]; H.CLIP['on']=0
for k, ident in enumerate(ids):
    if k % nparts != part: continue
    idx=int(man.index[man.id==ident][0]); e=man.loc[idx]
    d=gm.simulate_hmm6(int(e.n),600,tab[e.scenario],int(e.seed))
    y=np.array([float('%.12g'%v) for v in d['return'].values]); nt=int(e.n_train)
    mseed=(20261028+idx*15427+mindex*653)%(2**32)
    fit=comp.fit_model('EGARCH11-t',y[:nt],starts=5,maxiter=320,seed=mseed)
    p=np.array([fit['params'][q] for q in fit['parameter_names']])
    ls,h,inn=comp.eval_model('EGARCH11-t',y,p,fit['training_initvar'])
    pen=int((h[nt:]<=1e-20).any()); sc=float(np.mean(ls[nt:]))
    hmm=float(rec.loc[ident,'oos_mean_logscore'])
    rows.append(dict(id=ident,scenario=e.scenario,egarch_score=sc,penalized=pen,hmm_oos=hmm,hmm_minus_egarch=hmm-sc,
                     alpha=p[2],beta=p[3],nu=p[-1],train_loglik=fit['train_loglik'],success=fit['optimizer_success']))
    if len(rows)%25==0: pd.DataFrame(rows).to_csv(outfn,index=False)
pd.DataFrame(rows).to_csv(outfn,index=False)
