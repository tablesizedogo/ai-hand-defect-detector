import json,glob,numpy as np
from sklearn.metrics import roc_auc_score
w=json.load(open('auto/weak40.json'))
import sys
vl={}
for f in glob.glob('auto/vlm_*.json'):
    t=f.split('vlm_')[1][:-5]
    if t in('smoke','dbg'): continue
    d=json.load(open(f)); vl[t]=d
ks=[k for k in w if w[k]['gold']!='nohand']; y=np.array([w[k]['gold']=='ok' for k in ks]).astype(int)
print('ok',y.sum(),'bad',(1-y).sum())
def rep(name,s,thr=None):
    s=np.array(s,float); auc=roc_auc_score(y,s)
    # best balanced accuracy threshold (optimistic) and recall_bad at thr
    best=max(((((s>=t)&(y==1)).sum()/y.sum()+((s<t)&(y==0)).sum()/(1-y).sum())/2,t) for t in np.unique(s))
    line=f'{name:28s} AUC={auc:.2f} bestBalAcc={best[0]:.2f}@{best[1]:.3g}'
    if thr is not None:
        p=s>=thr; acc=(p==y).mean(); rb=((~p)&(y==0)).sum()/(1-y).sum(); fa=((~p)&(y==1)).sum()/y.sum()
        line+=f' | thr={thr}: acc={acc:.2f} recall_bad={rb:.2f} false_alarm_ok={fa:.2f}'
    print(line)
rep('ADetailer yolo conf',[w[k]['adetailer_conf'] for k in ks])
rep('MediaPipe detected',[w[k]['mp_detected']+0.01*w[k]['mp_score'] for k in ks],0.5)
rep('MediaPipe -geo_z (det only)',[-(w[k]['mp_geo_z'] if w[k]['mp_geo_z'] is not None else 50) for k in ks])
rep('CLIP B32 zero-shot p_ok',[w[k]['clip_p_ok'] for k in ks],0.5)
rep('HandCraft YOLO p_std',[w[k]['hc_p_ok'] for k in ks],0.5)
rep('HandCraft YOLO -bad conf',[-w[k]['hc_bad'] for k in ks])
rep('CountHallu quality cls',[w[k]['counthallu_q'] for k in ks],0.5)
for t,d in vl.items():
    kk=[k for k in ks if k in d]
    if len(kk)<len(ks): print(t,'partial',len(kk)); 
    yy=np.array([w[k]['gold']=='ok' for k in kk]).astype(int)
    s=np.array([d[k]['score'] for k in kk])
    if len(kk)==len(ks): rep('VLM '+t,s,3.0)
    print(f'VLM {t:20s} n={len(kk)} AUC={roc_auc_score(yy,s):.2f} mean_score ok={s[yy==1].mean():.2f} bad={s[yy==0].mean():.2f} sec/crop={np.median([d[k]["sec"] for k in kk]):.1f}')
    nh=[k for k in w if w[k]['gold']=='nohand' and k in d]; print('   p_none on nohand',[round(d[k]['p_none'],2) for k in nh],' on hands mean',round(np.mean([d[k]['p_none'] for k in kk]),2))
nh=[k for k in w if w[k]['gold']=='nohand']
print('nohand: clip_p_nohand',[round(w[k]['clip_p_nohand'],2) for k in nh],'hands mean',round(np.mean([w[k]['clip_p_nohand'] for k in ks]),2),' mp_det',[w[k]['mp_detected'] for k in nh])
