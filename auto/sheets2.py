import json,sys
from pathlib import Path
from PIL import Image, ImageDraw
R=Path('/workspace/hand_defects'); keys=json.load(open(R/'auto/shuffled.json'))
T=256
for s in range(int(sys.argv[1]),int(sys.argv[2])):
    sheet=Image.new('RGB',(6*T,4*T),(20,20,20)); d=ImageDraw.Draw(sheet)
    for i,k in enumerate(keys[s*24:(s+1)*24]):
        c=Image.open(R/f"auto/crops/{k}.jpg"); f=(T-4)/max(c.size); c=c.resize((int(c.width*f),int(c.height*f)),Image.LANCZOS)
        x,y=(i%6)*T,(i//6)*T; sheet.paste(c,(x,y)); d.rectangle([x,y,x+40,y+14],fill=(0,0,0)); d.text((x+2,y+1),str(s*24+i),fill=(255,255,0))
    sheet.save(R/f'auto/sheets/t{s:03d}.jpg',quality=88)
