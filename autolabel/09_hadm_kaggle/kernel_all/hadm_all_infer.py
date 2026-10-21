"""Kaggle kernel (GPU T4 x2, internet on, private): HADM-L on all ~2,500 source images behind the 5,604 crops.
Input dataset artem4767/hand-defects-hadm-all-input (images/, boxes.json, hadm_infer.py). Two shards, one per T4
(the second process starts after the first has loaded the checkpoint, to keep peak RAM low). Output
/kaggle/working/hadm_detections.json (merged) + per-shard files + hadm_run.json."""
import glob, json, os, subprocess, sys, time
T0 = time.time(); timings = {}
def sh(c):
    print('+', c, flush=True); t = time.time(); subprocess.run(c, shell=True, check=True); return time.time() - t
sh('nvidia-smi || true')
import torch
ngpu = torch.cuda.device_count(); print('torch', torch.__version__, 'gpus', ngpu, flush=True)
if ngpu == 0: raise SystemExit('NO GPU in this session')
timings['pip'] = sh(f'{sys.executable} -m pip install -q fvcore iopath omegaconf hydra-core pycocotools && '
                    f'{sys.executable} -m pip install -q --no-deps fairscale')
HADM_COMMIT = 'd09f66758e169a9271efadf6233c7bb912eb1a9e'
timings['clone'] = sh(f'git clone -q https://github.com/wangkaihong/HADM.git /tmp/HADM && git -C /tmp/HADM checkout -q {HADM_COMMIT}')
URL = 'https://www.dropbox.com/scl/fi/zwasvod906x1akzinnj3i/HADM-L_0249999.pth?rlkey=bqz5517tm8yt8l6ngzne4xejx&st=k1a1gzph&dl=1'
W = '/tmp/HADM-L_0249999.pth'
timings['weights_download'] = sh(f'curl -sSL --retry 3 -o {W} "{URL}" && ls -l {W}')
assert os.path.getsize(W) == 5733865413, os.path.getsize(W)
inp = os.path.dirname(glob.glob('/kaggle/input/**/hadm_infer.py', recursive=True)[0]); print('input', inp, flush=True)
n = min(ngpu, 2); procs = []; t = time.time()
for s in range(n):
    log = open(f'/kaggle/working/shard{s}.log', 'w')
    procs.append(subprocess.Popen(f'CUDA_VISIBLE_DEVICES={s} {sys.executable} {inp}/hadm_infer.py --repo /tmp/HADM --weights {W} '
                                  f'--images {inp}/images --out /kaggle/working/det_shard{s}.json --score-thresh 0.001 --shard {s} --nshards {n}',
                                  shell=True, stdout=log, stderr=subprocess.STDOUT))
    if s < n - 1:  # wait for this shard's model to be loaded before starting the next one
        while 'model ready' not in open(f'/kaggle/working/shard{s}.log').read() and procs[-1].poll() is None: time.sleep(5)
last = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(30)
    if time.time() - last > 300:
        for s in range(n): print(f'shard{s}:', open(f'/kaggle/working/shard{s}.log').read().strip().splitlines()[-1][:120], flush=True)
        last = time.time()
for s, p in enumerate(procs):
    print(f'--- shard{s} rc={p.returncode}', flush=True); print(''.join(open(f'/kaggle/working/shard{s}.log').readlines()[-5:]), flush=True)
    assert p.returncode == 0
timings['inference_total'] = time.time() - t
parts = [json.load(open(f'/kaggle/working/det_shard{s}.json')) for s in range(n)]
det = {}; [det.update(p['detections']) for p in parts]
json.dump(dict(meta=dict(parts[0]['meta'], n_images=len(det), shards=[p['meta'] for p in parts]), detections=det), open('/kaggle/working/hadm_detections.json', 'w'))
os.remove(W); timings['kernel_total'] = time.time() - T0
json.dump(dict(timings=timings, hadm_commit=HADM_COMMIT, torch=torch.__version__, n_gpus_used=n,
               gpu=torch.cuda.get_device_name(0)), open('/kaggle/working/hadm_run.json', 'w'), indent=1)
print('DONE', len(det), timings, flush=True)
