import json,numpy as np, mediapipe as mp
from pathlib import Path
from PIL import Image
from mediapipe.tasks.python import vision, BaseOptions
R=Path('/workspace/hand_defects')
opt=vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(R/'weights/hand_landmarker.task')),num_hands=2,min_hand_detection_confidence=0.1,min_hand_presence_confidence=0.1)
det=vision.HandLandmarker.create_from_options(opt)
crops=json.load(open(R/'auto/crops.json'))
BONES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
out={}
for c in crops:
    im=Image.open(R/f"auto/crops/{c['key']}.jpg").convert('RGB'); f=384/max(im.size); im=im.resize((int(im.width*f),int(im.height*f)))
    r=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.asarray(im)))
    if not r.hand_world_landmarks: out[c['key']]=dict(n=0); continue
    i=int(np.argmax([h[0].score for h in r.handedness]))
    W=np.array([[p.x,p.y,p.z] for p in r.hand_world_landmarks[i]])
    L=np.array([np.linalg.norm(W[a]-W[b]) for a,b in BONES]); L=L/L[8]  # normalize by wrist->middle MCP
    out[c['key']]=dict(n=len(r.handedness),score=float(r.handedness[i][0].score),bones=L.tolist())
# z-score of bone ratios across detected hands
ks=[k for k in out if out[k]['n']]; B=np.array([out[k]['bones'] for k in ks]); med=np.median(B,0); mad=np.median(abs(B-med),0)+1e-6
for k,b in zip(ks,B): out[k]['geo_z']=float(np.max(abs(b-med)/mad))
json.dump(out,open(R/'auto/mp.json','w'))
print('detected',len(ks),'of',len(out))
