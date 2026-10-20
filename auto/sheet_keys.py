import json,sys
from pathlib import Path
from PIL import Image, ImageDraw
R=Path('/workspace/hand_defects'); keys=json.load(open(sys.argv[1])); out=sys.argv[2]; T=256
sheet=Image.new('RGB',(6*T,4*T),(20,20,20)); d=ImageDraw.Draw(sheet)
for i,k in enumerate(keys[:24]):
    c=Image.open(R/f"auto/crops/{k}.jpg"); f=(T-4)/max(c.size); c=c.resize((int(c.width*f),int(c.height*f)),Image.LANCZOS)
    x,y=(i%6)*T,(i//6)*T; sheet.paste(c,(x,y)); d.rectangle([x,y,x+28,y+14],fill=(0,0,0)); d.text((x+2,y+1),str(i),fill=(255,255,0))
sheet.save(out,quality=88)
