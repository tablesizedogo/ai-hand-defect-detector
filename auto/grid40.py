import json,glob,os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
R=Path('/workspace/hand_defects'); w=json.load(open(R/'auto/weak40.json'))
vl={t:json.load(open(R/f'auto/vlm_{t}.json')) for t in ['qwen3b','qwen7b','gemma4e4b','gemma4e4b_yn','qwen7b_yn'] if os.path.exists(R/f'auto/vlm_{t}.json')}
order=sorted(w,key=lambda k:{'bad':0,'ok':1,'nohand':2}[w[k]['gold']])
T=230; TH=100; cols=8; rows=(len(order)+cols-1)//cols
try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
except: font=ImageFont.load_default()
sheet=Image.new('RGB',(cols*T,rows*(T+TH)+30),(18,18,18)); d=ImageDraw.Draw(sheet)
d.text((6,6),'Gold (my visual judgment): red=BAD, green=OK, grey=NO HAND. Scores: higher = more likely OK. VLM = 1-5 rating expectation.',fill=(255,255,255),font=font)
col={'bad':(239,68,68),'ok':(34,197,94),'nohand':(150,150,150)}
for i,k in enumerate(order):
    x=(i%cols)*T; y=30+(i//cols)*(T+TH); r=w[k]
    c=Image.open(R/f'auto/crops/{k}.jpg').convert('RGB'); f=(T-10)/max(c.size); c=c.resize((int(c.width*f),int(c.height*f)))
    d.rectangle([x+1,y+1,x+T-2,y+T-2],outline=col[r['gold']],width=4); sheet.paste(c,(x+5,y+5))
    lines=[f"#{i} {r['gold'].upper()}  {k}",f"CLIP p_ok {r['clip_p_ok']:.2f}  ADet {r['adetailer_conf']:.2f}",
           f"MediaPipe {'det' if r['mp_detected'] else 'none'}  HandCraft {r['hc_p_ok']:.2f}",f"CountHallu {r['counthallu_q']:.2f}"]
    tl=[f"{t.replace('gemma4e4b','g4').replace('qwen','q')} {vl[t][k]['score']:.1f}" for t in vl if k in vl[t]]
    lines.append('VLM '+'  '.join(tl[:3])); lines.append('    '+'  '.join(tl[3:]))
    for j,l in enumerate(lines): d.text((x+4,y+T+3+j*15),l,fill=(230,230,230),font=font)
sheet.save(R/'auto/test40_grid.jpg',quality=88); print(sheet.size)
