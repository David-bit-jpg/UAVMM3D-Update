import sys,pickle,json,os
sys.argv=['x']; sys.path.insert(0,'.')
import numpy as np, make_fig_qual_cmp4_1117 as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
C=F.C; BL=('monocop','monodgp','centernet')
CFG={'mav':[('mavic2',0.25,1.6),('phantom4',0.15,2.0)],'anti':[('near',0.30,1.3),('far',0.30,1.3)]}
cand={}
for ds in ('mav','anti'):
    cp='cache_%s.pkl'%ds
    if os.path.exists(cp): M,R=pickle.load(open(cp,'rb'))
    else:
        R={}
        for key,_,_,runs in F.METHODS:
            M,pb,d,e,rng_,tp,aoe=C.pf(ds,runs[ds]); R[key]=(pb,d,rng_,tp,aoe)
        M={k:M[k] for k in M.files}; pickle.dump((M,R),open(cp,'wb'))
    K=M['K']; gt=M['gt']; N=len(gt); rng_=R['pst'][2]
    T=np.where(np.all([R[k][3] for k in R],0))[0]
    vis={}
    for k in R:
        v=np.full(N,np.inf)
        for i in T: v[i]=C.corner_err(R[k][0][i],gt[i],K[i])
        vis[k]=v
    base=np.minimum.reduce([vis[k] for k in BL])
    cls=np.array([str(c) for c in M['cls']]); seq=np.array([str(x) for x in M['seq']]); frm=np.array([str(x) for x in M['frame']])
    med=np.median(rng_[T]); cand[ds]=[]
    for name,q,ratio in CFG[ds]:
        g=T[cls[T]==name] if ds=='mav' else (T[rng_[T]<med] if name=='near' else T[rng_[T]>=med])
        g=g[(vis['pst'][g]<=np.percentile(vis['pst'][g],100*q))&(base[g]>=ratio*vis['pst'][g])]
        o=sorted(g,key=lambda j:-(base[j]-2*vis['pst'][j])); used=set(); pk=[]
        for i in o:
            kk=seq[i] if ds=='mav' else int(frm[i][:6])//300
            if kk in used: continue
            used.add(kk); pk.append((seq[i],frm[i]))
            print(ds,name,seq[i],frm[i],'rng %.1f pst %.2f/%.3f |'%(rng_[i],vis['pst'][i],R['pst'][1][i]),' '.join('%s %.2f/%.2f'%(k,vis[k][i],R[k][1][i]) for k in BL))
            if len(pk)==4: break
        print(ds,name,'pool',len(g)); cand[ds].append(pk)
json.dump(cand,open('cand_best3.json','w'))
metas={}
for ds,p in F.MF.CACHE.items():
    I=pickle.load(open(p,'rb')); metas[ds]={(m['seq'],m['frame']):m for m in I['metas']}
for ds in ('mav','anti'):
    for gi,pk in enumerate(cand[ds]):
        if not pk: continue
        C.FIXED={'mav':[],'anti':[]}; C.FIXED[ds]=[tuple(x) for x in pk]
        cols=[c for c in F.collect() if c['ds']==ds]
        fig=plt.figure(figsize=(3*len(cols),7))
        for k,c in enumerate(cols):
            meta=metas[ds][(c['seq'],c['frame'])]; img=F.MF.imread(meta)[:,:,::-1]; K,win=F.window(c,meta)
            for row,(key,name,col,_) in enumerate(F.METHODS):
                ax=fig.add_subplot(4,len(cols),row*len(cols)+k+1); F.panel(ax,img,K,win,c,key,col,'%d %s'%(k,c['frame'][:10]) if row==0 else None)
        fig.savefig('candB_%s%d.png'%(ds,gi),dpi=70); plt.close(fig)
