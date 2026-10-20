"""Step 3: resumable VLM scoring of every crop with Gemma-4-E4B (GGUF, llama.cpp, CPU).

One neutral prompt per crop (identical to the feasibility study) answered with one token: Yes / No / None.
From the first-token logprobs we keep p_yes, p_no, p_none:
  p_ok   = p_yes / (p_yes + p_no)              -> "hand is anatomically OK"
  p_hand = 1 - p_none / (p_yes + p_no + p_none) -> "this crop shows a human hand"
Results are appended to autolabel_data/vlm_gemma4e4b.jsonl (one line per crop); already-done keys are skipped on restart.
"""
import base64, io, json, math, os, sys, time
from pathlib import Path
from PIL import Image
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MTMDChatHandler

R = Path(__file__).resolve().parents[1]; D = R / 'autolabel_data'
MODEL = os.environ.get('VLM_MODEL', '/workspace/vlm/gemma-4-E4B-it-Q4_0.gguf')
MMPROJ = os.environ.get('VLM_MMPROJ', '/workspace/vlm/mmproj-gemma-4-E4B-it-Q8_0.gguf')
OUTF = D / os.environ.get('VLM_OUT', 'vlm_gemma4e4b.jsonl')
THREADS = int(os.environ.get('VLM_THREADS', '8')); SIDE = int(os.environ.get('VLM_SIDE', '448'))
PROMPT = ("Look at this cropped image region from an AI-generated picture. Judge only the hand(s). "
          "Is the hand anatomically correct (normal number of fingers, no fused/extra/missing fingers, no melted or broken shape)? "
          "Answer with exactly one word: Yes if the hand looks correct, No if it is malformed, None if there is no human hand.")

def log(msg):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), msg, flush=True)

keys = [json.loads(l)['key'] for l in open(D / 'manifest.jsonl')]
done = set()
if OUTF.exists():
    for l in open(OUTF):
        try: done.add(json.loads(l)['key'])
        except Exception: pass  # tolerate a truncated last line after a crash
todo = [k for k in keys if k not in done]
log(f'start: total={len(keys)} done={len(done)} todo={len(todo)} threads={THREADS} model={Path(MODEL).name}')
if not todo: log('ALL DONE'); sys.exit(0)

llm = Llama(model_path=MODEL, chat_handler=MTMDChatHandler(clip_model_path=MMPROJ, verbose=False),
            n_ctx=4096, n_threads=THREADS, verbose=False, logits_all=True)
t0 = time.time(); n = 0
with open(OUTF, 'a') as f:
    for k in todo:
        im = Image.open(D / f'crops/{k}.jpg').convert('RGB'); s = SIDE / max(im.size)
        im = im.resize((max(28, int(im.width * s)), max(28, int(im.height * s))), Image.BICUBIC)
        bio = io.BytesIO(); im.save(bio, 'JPEG', quality=92)
        url = 'data:image/jpeg;base64,' + base64.b64encode(bio.getvalue()).decode()
        t = time.time()
        try:
            r = llm.create_chat_completion(messages=[{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': url}}, {'type': 'text', 'text': PROMPT}]}],
                max_tokens=1, temperature=0, logprobs=True, top_logprobs=10)
            ch = r['choices'][0]; lp = {}
            for d in ch['logprobs']['content'][0]['top_logprobs']:
                tok = d['token'].strip().lower(); lp[tok] = max(lp.get(tok, -99.0), float(d['logprob']))
            py, pn, pz = (math.exp(lp.get(x, -30.0)) for x in ('yes', 'no', 'none'))
            rec = dict(key=k, ans=ch['message']['content'].strip(), p_yes=py, p_no=pn, p_none=pz,
                       p_ok=py / (py + pn + 1e-12), p_hand=1 - pz / (py + pn + pz + 1e-12), sec=round(time.time() - t, 2))
        except Exception as e:
            rec = dict(key=k, error=repr(e)[:300], sec=round(time.time() - t, 2))
        f.write(json.dumps(rec) + '\n'); f.flush(); n += 1
        if n % 25 == 0 or n == len(todo):
            rate = (time.time() - t0) / n; left = len(todo) - n
            log(f'progress: {len(done) + n}/{len(keys)} ({100 * (len(done) + n) / len(keys):.1f}%) '
                f'{rate:.2f} s/crop, ETA {left * rate / 3600:.2f} h')
log('ALL DONE')
