import json, random
from pathlib import Path
from PIL import Image, ImageDraw
R=Path('/workspace/hand_defects'); IMG=Path('/workspace/rag_civitai')
recs=[json.loads(l) for l in open(R/'data/preannotations.jsonl')]
out=[]
for r in recs:
    im=None
    for j,b in enumerate(r['boxes']):
        x0,y0,x1,y1=b['xyxy']; w,h=x1-x0,y1-y0
        if min(w,h)<56: continue
        if im is None: im=Image.open(IMG/r['path']).convert('RGB')
        s=max(w,h)*1.3/2; cx,cy=(x0+x1)/2,(y0+y1)/2
        box=[int(max(0,cx-s)),int(max(0,cy-s)),int(min(im.width,cx+s)),int(min(im.height,cy+s))]
        c=im.crop(box); c.thumbnail((336,336))
        key=f"{r['id']}_{j}"; c.save(R/f'auto/crops/{key}.jpg',quality=92)
        out.append(dict(key=key,id=r['id'],j=j,family=r['family'],style=r['style'],score=b['score'],side=min(w,h),hint=b.get('defect_hint')))
json.dump(out,open(R/'auto/crops.json','w'))
print(len(out))
random.seed(0); random.shuffle(out)
json.dump([o['key'] for o in out],open(R/'auto/shuffled.json','w'))
T=200
for s in range(12):
    sheet=Image.new('RGB',(8*T,5*T+0),(20,20,20)); d=ImageDraw.Draw(sheet)
    for i,o in enumerate(out[s*40:(s+1)*40]):
        c=Image.open(R/f"auto/crops/{o['key']}.jpg"); c.thumbnail((T,T))
        x,y=(i%8)*T,(i//8)*T; sheet.paste(c,(x,y)); d.rectangle([x,y,x+38,y+14],fill=(0,0,0)); d.text((x+2,y+1),str(s*40+i),fill=(255,255,0))
    sheet.save(R/f'auto/sheets/s{s:02d}.jpg',quality=88)
