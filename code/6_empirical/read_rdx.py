"""Minimal reader for R XDR serialization (RDX2) sufficient for a data.frame of numeric/int/character/Date columns."""
import struct, sys, numpy as np, pandas as pd
class R:
    def __init__(s,b): s.b=b; s.p=0; s.refs=[]
    def i(s): v=struct.unpack('>i',s.b[s.p:s.p+4])[0]; s.p+=4; return v
    def item(s):
        f=s.i(); t=f&0xFF; has_attr=f&(1<<9); has_tag=f&(1<<10)
        if t==254: return None
        if t in (253,242,251): return None   # global/empty/base env markers
        if t==255:
            idx=f>>8
            if idx==0: idx=s.i()
            return s.refs[idx-1]
        if t==1:
            v=s.item(); s.refs.append(('sym',v)); return ('sym',v)
        if t==2:   # pairlist
            out=[]
            while True:
                attr=s.item() if has_attr else None
                tag=s.item() if has_tag else None
                car=s.item(); out.append((tag[1] if tag else None,car))
                f=s.i(); t=f&0xFF; has_attr=f&(1<<9); has_tag=f&(1<<10)
                if t==254: return out
                if t!=2: raise ValueError('unexpected cdr type %d'%t)
        if t==9:
            n=s.i()
            if n==-1: return None
            v=s.b[s.p:s.p+n].decode('utf-8','replace'); s.p+=n; return v
        if t in (10,13):
            n=s.i(); v=np.frombuffer(s.b,'>i4',n,s.p).astype(float); s.p+=4*n; v[v==-2147483648]=np.nan; val=v
        elif t==14:
            n=s.i(); val=np.frombuffer(s.b,'>f8',n,s.p).astype(float); s.p+=8*n
        elif t==16:
            n=s.i(); val=[s.item() for _ in range(n)]
        elif t==19:
            n=s.i(); val=[s.item() for _ in range(n)]
        else: raise ValueError('unsupported SEXP type %d at %d'%(t,s.p))
        attrs=dict(s.item()) if has_attr else {}
        return ('vec',val,attrs)
def load(path):
    b=open(path,'rb').read(); assert b[:2]==b'RD'
    r=R(b); r.p=b.index(b'X\n')+2; r.i(); r.i(); r.i()
    top=r.item(); return dict(top)
if __name__=='__main__':
    obj=load(sys.argv[1]); name=list(obj)[0]; _,cols,attrs=obj[name]
    names=attrs['names'][1]
    data={}
    for nm,c in zip(names,cols):
        v=c[1]; a=c[2]
        if 'levels' in a: lev=a['levels'][1]; v=[lev[int(k)-1] for k in v]
        elif 'class' in a and 'Date' in a['class'][1]: v=pd.to_datetime(np.asarray(v),unit='D',origin='1970-01-01')
        data[nm]=v
    df=pd.DataFrame(data); print(df.head(3)); print(df.tail(3)); print(df.dtypes.to_dict(), len(df))
    df.to_csv(sys.argv[2],index=False)
