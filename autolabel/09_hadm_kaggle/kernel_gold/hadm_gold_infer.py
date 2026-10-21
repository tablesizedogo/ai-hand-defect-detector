"""Kaggle kernel (GPU T4, internet on, private): run HADM-L on the gold-set source images.
Input dataset artem4767/hand-defects-hadm-gold-input (images/, boxes.json, hadm_infer.py).
Output /kaggle/working/hadm_detections.json (+ hadm_run.json with timings / versions)."""
import glob, json, os, subprocess, sys, time
T0 = time.time(); timings = {}
def sh(c):
    print('+', c, flush=True); t = time.time(); subprocess.run(c, shell=True, check=True); return time.time() - t
sh('nvidia-smi || true')
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available(): raise SystemExit('NO GPU in this session')
timings['pip'] = sh(f'{sys.executable} -m pip install -q fvcore iopath omegaconf hydra-core pycocotools && '
                    f'{sys.executable} -m pip install -q --no-deps fairscale')
HADM_COMMIT = 'd09f66758e169a9271efadf6233c7bb912eb1a9e'
timings['clone'] = sh(f'git clone -q https://github.com/wangkaihong/HADM.git /tmp/HADM && git -C /tmp/HADM checkout -q {HADM_COMMIT}')
URL = 'https://www.dropbox.com/scl/fi/zwasvod906x1akzinnj3i/HADM-L_0249999.pth?rlkey=bqz5517tm8yt8l6ngzne4xejx&st=k1a1gzph&dl=1'
W = '/tmp/HADM-L_0249999.pth'
timings['weights_download'] = sh(f'curl -sSL --retry 3 -o {W} "{URL}" && ls -l {W}')
size = os.path.getsize(W); print('weights bytes', size, flush=True)
assert size == 5733865413, f'unexpected weights size {size}'
inp = os.path.dirname(glob.glob('/kaggle/input/**/hadm_infer.py', recursive=True)[0]); print('input', inp, flush=True)
timings['inference_total'] = sh(f'{sys.executable} {inp}/hadm_infer.py --repo /tmp/HADM --weights {W} '
                                f'--images {inp}/images --out /kaggle/working/hadm_detections.json --score-thresh 0.001')
os.remove(W)
timings['kernel_total'] = time.time() - T0
json.dump(dict(timings=timings, hadm_commit=HADM_COMMIT, torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0)), open('/kaggle/working/hadm_run.json', 'w'), indent=1)
print('DONE', timings, flush=True)
