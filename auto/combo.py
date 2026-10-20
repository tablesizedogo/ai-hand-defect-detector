import json,os,numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
w=json.load(open('auto/weak40.json')); mined=set(json.load(open('auto/clip_mined_bad.json')))
vl={t:json.load(open(f'auto/vlm_{t}.json')) for t in ['qwen3b','qwen7b','gemma4e4b','gemma4e4b_yn','qwen7b_yn'] if os.path.exists(f'auto/vlm_{t}.json')}
ks=[k for k in w if w[k]['gold']!='nohand']; y=np.array([w[k]['gold']=='ok' for k in ks]).astype(int); sub=np.array([k not in mined for k in ks])
S={'ADetailer conf':[w[k]['adetailer_conf'] for k in ks],'CLIP zs':[w[k]['clip_p_ok'] for k in ks],'MediaPipe det':[w[k]['mp_detected']+.01*w[k]['mp_score'] for k in ks]}
for t in vl: S['VLM '+t]=[vl[t][k]['score'] for k in ks]
R={n:rankdata(v)/len(v) for n,v in S.items()}
if 'VLM gemma4e4b_yn' in R:
    S['gemma_yn+CLIP+ADet (rank avg)']=R['VLM gemma4e4b_yn']+R['CLIP zs']+R['ADetailer conf']
    S['gemma_yn+ADet (rank avg)']=R['VLM gemma4e4b_yn']+R['ADetailer conf']
if 'VLM qwen7b_yn' in R: S['qwen7b_yn+gemma_yn+ADet']=R['VLM qwen7b_yn']+R['VLM gemma4e4b_yn']+R['ADetailer conf']
for n,v in S.items():
    v=np.array(v,float); print(f'{n:34s} AUC all={roc_auc_score(y,v):.2f}  AUC excl. CLIP-mined bads={roc_auc_score(y[sub],v[sub]):.2f}')
