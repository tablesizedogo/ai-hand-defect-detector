import json,sys,time,base64,re,math
from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler, MTMDChatHandler
R=Path('/workspace/hand_defects')
model,mmproj,tag=sys.argv[1],sys.argv[2],sys.argv[3]
keys=list(json.load(open(R/'auto/gold40.json'))) if len(sys.argv)<5 else sys.argv[4].split(',')
H=Qwen25VLChatHandler(clip_model_path=mmproj,verbose=False) if False else MTMDChatHandler(clip_model_path=mmproj,verbose=False)
llm=Llama(model_path=model,chat_handler=H,n_ctx=4096,n_threads=8,verbose=False,logits_all=True)
P0=("Look at this cropped image region from an AI-generated picture. Judge only the hand(s). "
   "Is the hand anatomically correct (normal number of fingers, no fused/extra/missing fingers, no melted or broken shape)? "
   "Answer with exactly one word: Yes if the hand looks correct, No if it is malformed, None if there is no human hand.")
P=("This is a crop from an AI-generated image. AI images often have malformed hands: extra or missing fingers, fused fingers, "
   "wrong joints, melted or blob-like shapes, hands merging with objects. Inspect the hand carefully and count the fingers. "
   "Rate the anatomical correctness of the hand from 1 to 5 (1 = clearly malformed, 3 = unsure, 5 = clearly correct), or 0 if there is no human hand. "
   "Reply with a single digit.")
if tag.endswith('_yn'): P=P0
out={}
for k in keys:
    from PIL import Image; import io
    im=Image.open(R/f'auto/crops/{k}.jpg').convert('RGB'); f=448/max(im.size); im=im.resize((max(28,int(im.width*f)),max(28,int(im.height*f))),Image.BICUBIC)
    bio=io.BytesIO(); im.save(bio,'JPEG',quality=92); b64=base64.b64encode(bio.getvalue()).decode()
    t=time.time()
    r=llm.create_chat_completion(messages=[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+b64}},{"type":"text","text":P}]}],
        max_tokens=2,temperature=0,logprobs=True,top_logprobs=10)
    dt=time.time()-t
    ch=r['choices'][0]; txt=ch['message']['content'].strip()
    lp={}
    try:
        for d in ch['logprobs']['content'][0]['top_logprobs']: lp[d['token'].strip().lower()]=max(lp.get(d['token'].strip().lower(),-99),d['logprob'])
    except Exception as e: pass
    pd={d:math.exp(lp.get(str(d),-30)) for d in range(6)}; Z=sum(pd.values())+1e-12
    score=sum(d*pd[d] for d in range(1,6))/(sum(pd[d] for d in range(1,6))+1e-12)
    py=math.exp(lp.get('yes',-30)); pn=math.exp(lp.get('no',-30)); pz=math.exp(lp.get('none',-30))
    out[k]=dict(ans=txt,score=(py/(py+pn+1e-12)*4+1) if tag.endswith('_yn') else score,p_yes=py,p_no=pn,p_ok=(score-1)/4,p_none=(pz/(py+pn+pz+1e-12)) if tag.endswith('_yn') else pd[0]/Z,sec=round(dt,2))
    print(k,out[k],[(d["token"],round(d["logprob"],2)) for d in ch["logprobs"]["content"][0]["top_logprobs"]][:6],flush=True)
json.dump(out,open(R/f'auto/vlm_{tag}.json','w'),indent=1)
