import json,numpy as np,torch,timm
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
R=Path('/workspace/hand_defects'); torch.set_num_threads(4)
gold=json.load(open(R/'auto/gold40.json')); crops={c['key']:c for c in json.load(open(R/'auto/crops.json'))}
mp=json.load(open(R/'auto/mp.json')); cz=json.load(open(R/'auto/clip_zs.json'))
yc=YOLO(str(R/'weights/ext/handcraft_yolo.pt'))
cq=timm.create_model('maxvit_small_tf_224',pretrained=False,num_classes=2); cq.load_state_dict(torch.load(R/'weights/ext/counthallu_quality.pth',map_location='cpu')); cq.eval()
import torchvision.transforms as T
tf=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
res={}
for k in gold:
    im=Image.open(R/f'auto/crops/{k}.jpg').convert('RGB')
    r=yc.predict(im.resize((512,int(512*im.height/im.width))) if im.width>=im.height else im.resize((int(512*im.width/im.height),512)),imgsz=512,conf=0.05,verbose=False)[0]
    cls=r.boxes.cls.numpy(); cf=r.boxes.conf.numpy()
    s_ok=float(cf[cls==0].max()) if (cls==0).any() else 0.; s_bad=float(cf[cls==1].max()) if (cls==1).any() else 0.
    with torch.no_grad(): q=torch.softmax(cq(tf(im)[None]),1)[0,1].item()
    m=mp[k]
    res[k]=dict(gold=gold[k],adetailer_conf=crops[k]['score'],mp_detected=int(m['n']>0),mp_score=m.get('score',0.),mp_geo_z=m.get('geo_z'),
        clip_p_ok=cz[k]['p_ok'],clip_p_nohand=cz[k]['p_nohand'],hc_ok=s_ok,hc_bad=s_bad,hc_p_ok=(s_ok+1e-3)/(s_ok+s_bad+2e-3),counthallu_q=q)
json.dump(res,open(R/'auto/weak40.json','w'),indent=1)
print('ok')
