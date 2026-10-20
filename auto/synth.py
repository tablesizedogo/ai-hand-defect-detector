import json,numpy as np,cv2,mediapipe as mp
from pathlib import Path
from PIL import Image, ImageDraw
from mediapipe.tasks.python import vision, BaseOptions
R=Path('/workspace/hand_defects'); rng=np.random.default_rng(1)
det=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(R/'weights/hand_landmarker.task')),num_hands=1))
gold=json.load(open(R/'auto/gold40.json')); mpj=json.load(open(R/'auto/mp.json'))
crops=[c['key'] for c in json.load(open(R/'auto/crops.json'))]
cands=[k for k in crops if mpj[k]['n'] and mpj[k]['geo_z']<3 and mpj[k]['score']>0.9 and k not in gold][:300]
FING=[(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)]
def lm(img):
    r=det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=img))
    if not r.hand_landmarks: return None
    return np.array([[p.x*img.shape[1],p.y*img.shape[0]] for p in r.hand_landmarks[0]])
def finger_mask(img,P,f,w):
    m=np.zeros(img.shape[:2],np.uint8)
    for a,b in zip(f[:-1],f[1:]): cv2.line(m,tuple(P[a].astype(int)),tuple(P[b].astype(int)),255,int(w))
    cv2.circle(m,tuple(P[f[-1]].astype(int)),int(w/2),255,-1); return m
def extra_finger(img,P):
    f=FING[rng.integers(4)]; w=np.linalg.norm(P[5]-P[17])/2.6
    m=finger_mask(img,P,f,w); ang=rng.choice([-1,1])*rng.uniform(24,36)
    M=cv2.getRotationMatrix2D(tuple(P[f[0]].astype(float)),ang,1.0)
    fi=cv2.warpAffine(img,M,img.shape[1::-1],borderMode=cv2.BORDER_REFLECT); fm=cv2.warpAffine(m,M,img.shape[1::-1])
    fm=cv2.GaussianBlur(fm,(0,0),w/6)[...,None]/255.
    return (img*(1-fm)+fi*fm).astype(np.uint8)
def missing_finger(img,P):
    f=FING[rng.integers(1,3)]; w=np.linalg.norm(P[5]-P[17])/3.0
    m=finger_mask(img,P,(f[1],f[2],f[3]),w)
    return cv2.inpaint(img,m,7,cv2.INPAINT_TELEA)
def melt(img,P):
    h,w=img.shape[:2]; c=P[9]; s=np.linalg.norm(P[0]-P[9])
    dx=cv2.GaussianBlur((rng.standard_normal((h,w))).astype(np.float32),(0,0),s/4)*s*3.0
    dy=cv2.GaussianBlur((rng.standard_normal((h,w))).astype(np.float32),(0,0),s/4)*s*3.0
    yy,xx=np.mgrid[0:h,0:w].astype(np.float32); g=np.exp(-((xx-c[0])**2+(yy-c[1])**2)/(2*(s*0.9)**2))
    return cv2.remap(img,(xx+dx*g).astype(np.float32),(yy+dy*g).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
rows=[];n=0
for k in cands:
    img=np.asarray(Image.open(R/f'auto/crops/{k}.jpg').convert('RGB')); f=320/max(img.shape[:2]); img=cv2.resize(img,None,fx=f,fy=f)
    P=lm(img)
    if P is None: continue
    rows.append([img,extra_finger(img,P),missing_finger(img,P),melt(img,P)]); n+=1
    if n==4: break
T=330; sheet=Image.new('RGB',(4*T,len(rows)*T+20),(20,20,20)); d=ImageDraw.Draw(sheet)
for j,t in enumerate(['original','extra finger','missing finger','mesh melt']): d.text((j*T+4,4),t,fill=(255,255,0))
for i,r in enumerate(rows):
    for j,im in enumerate(r):
        p=Image.fromarray(im); p.thumbnail((T-4,T-4)); sheet.paste(p,(j*T,20+i*T))
sheet.save(R/'auto/synthetic_negatives_demo_v2.jpg',quality=90); print(n)
