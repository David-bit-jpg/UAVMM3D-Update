# Rescore every *test*.pkl prediction under the unified protocol (pack score.py as of 11-16, stable sort).
# Split chosen by key coverage against the two sealed TEST metas. Also reports default-argsort AP to flag tie sensitivity.
import sys, os, glob, pickle, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0,'D:/UAV3D_baselines_pack/scripts'); import score as S
METAS={'real_test':'D:/UAV3D_baselines_pack/data/gt/real_test_meta.npz','outdoor_test':'D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz'}
def ap_default(sc,tp,n_gt):
    if not len(sc): return 0.0
    o=np.argsort(-sc); tp=tp[o].astype(float); ctp,cfp=np.cumsum(tp),np.cumsum(1-tp)
    rec,prec=ctp/max(n_gt,1),ctp/np.maximum(ctp+cfp,1); rs=np.linspace(0,1,101)
    p=np.array([prec[rec>=r].max() if (rec>=r).any() else 0.0 for r in rs])[rs>0.1]
    return float(np.mean(np.clip(p-0.1,0,None))/0.9)
MM=None
def init():
    global MM; MM={k:np.load(v) for k,v in METAS.items()}
def one(f):
    try:
        R=pickle.load(open(f,'rb'))
        if not isinstance(R,list) or not R or not isinstance(R[0],dict) or 'boxes9d' not in R[0]: return f,{'err':'format'}
        best=None
        for split,M in MM.items():
            key={(str(s),str(fr)):i for i,(s,fr) in enumerate(zip(M['seq'],M['frame']))}
            hit=sum((str(r['seq']),str(r['frame'])) in key for r in R)
            if best is None or hit>best[2]: best=(split,key,hit)
        split,key,hit=best; M=MM[split]; N=len(key)
        if hit==0: return f,{'err':'nokeys'}
        P={k:np.full((N,3),np.nan) for k in ('pt','plwh','peul')}; P['conf']=np.full(N,np.nan)
        for r in R:
            i=key.get((str(r['seq']),str(r['frame']))); b=np.asarray(r['boxes9d'],float).reshape(-1,9); s=np.asarray(r['scores'],float).reshape(-1)
            if i is None or not len(b): continue
            j=int(np.argmax(s)); P['pt'][i],P['plwh'][i],P['peul'][i],P['conf'][i]=b[j,:3],b[j,3:6],b[j,6:9],s[j]
        m=S.metrics(P,M['gt'],M['K'],np.arange(N),S.CFG[split])
        old=S.nus_ap; S.nus_ap=ap_default
        try: m2=S.metrics(P,M['gt'],M['K'],np.arange(N),S.CFG[split])
        finally: S.nus_ap=old
        c=P['conf'][np.isfinite(P['conf'])]; u,cnt=np.unique(c,return_counts=True)
        out=dict(split=split,hit=hit,N=N,ties=int(cnt[cnt>1].sum()),AP_default=m2['AP'],UDS_default=m2['UDS'],**m)
        if split=='outdoor_test': out['ATE']*=100
        return f,out
    except Exception as e:
        return f,{'err':repr(e)[:200]}
if __name__=='__main__':
    roots=sys.argv[2:]; files=[]
    for r in roots:
        for f in glob.glob(r+'/**/*.pkl',recursive=True):
            b=os.path.basename(f)
            if 'test' in b and 'val' not in b and 'train' not in b: files.append(f.replace(os.sep,'/'))
    files=sorted(set(files)); print(len(files),'files',flush=True)
    with Pool(12,initializer=init) as p: res=dict(p.map(one,files,chunksize=2))
    json.dump(res,open(sys.argv[1],'w'),indent=0)
    print('done',sum('err' not in v for v in res.values()))
