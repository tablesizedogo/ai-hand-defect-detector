import json,torch,open_clip,numpy as np
from pathlib import Path
from PIL import Image
R=Path('/workspace/hand_defects'); torch.set_num_threads(8)
m,_,pre=open_clip.create_model_and_transforms('ViT-B-32',pretrained='laion2b_s34b_b79k'); tok=open_clip.get_tokenizer('ViT-B-32'); m.eval()
crops=json.load(open(R/'auto/crops.json'))
good=["a photo of a normal human hand with five fingers","a well-drawn hand with correct anatomy","a natural looking hand"]
bad=["a deformed hand with extra fingers","a malformed mutated hand with fused fingers","a distorted melted hand, bad anatomy"]
noh=["an image with no hands","an animal paw","a foot"]
with torch.no_grad():
    T=m.encode_text(tok(good+bad+noh)); T/=T.norm(dim=-1,keepdim=True)
    feats=[]
    for i in range(0,len(crops),64):
        x=torch.stack([pre(Image.open(R/f"auto/crops/{c['key']}.jpg").convert('RGB')) for c in crops[i:i+64]])
        f=m.encode_image(x); f/=f.norm(dim=-1,keepdim=True); feats.append(f)
    F=torch.cat(feats)
    L=100*F@T.T
    g=L[:,0:3].logsumexp(1); b=L[:,3:6].logsumexp(1)
    p_ok=torch.sigmoid(g-b).numpy()
    p_noh=torch.softmax(torch.stack([g,b,L[:,6:9].logsumexp(1)],1),1)[:,2].numpy()
np.save(R/'auto/clipb32_feats.npy',F.numpy())
json.dump({c['key']:dict(p_ok=float(p),p_nohand=float(q)) for c,p,q in zip(crops,p_ok,p_noh)},open(R/'auto/clip_zs.json','w'))
print('done',len(crops))
