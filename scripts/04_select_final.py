"""Stage 1c: pick ~2,500 images with at least one clearly visible hand, balanced across base-model families and
styles, max 3 images per author, no near-duplicates (CLIP image cosine >= 0.92)."""
import json, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ART = Path("/workspace/rag_civitai/search/artifacts")
QUOTA = {"krea2": 700, "illustrious": 700, "flux": 600, "extras": 500}
STYLES = ["photo", "anime", "art"]
MAX_RISK = 0.022  # CLIP fan-service/body-emphasis score (01_select_candidates.py); tuned by visual review
MANUAL = {int(x) for x in open(ROOT / "data/manual_exclude.txt").read().split() if x.isdigit()}
MIN_CONF, MIN_SIDE, MAX_BOXES, MAX_PER_USER, DUP_COS = 0.35, 40, 6, 3, 0.92

cands = {r["id"]: r for r in map(json.loads, open(ROOT / "data/candidates.jsonl"))}
dets = {d["id"]: d for d in map(json.loads, open(ROOT / "data/det_v8s.jsonl"))}
reasons = Counter(); pool = []
for iid, r in cands.items():
    d = dets.get(iid)
    if d is None: reasons["not_detected_yet"] += 1; continue
    good = [b for b in d["boxes"] if b[4] >= MIN_CONF and min(b[2] - b[0], b[3] - b[1]) >= MIN_SIDE]
    if not good: reasons["no_clear_hand"] += 1; continue
    if len([b for b in d["boxes"] if b[4] >= 0.25]) > MAX_BOXES: reasons["too_many_hands"] += 1; continue
    r["n_hands_screen"] = len(good); r["best_hand_conf"] = max(b[4] for b in good)
    pool.append(r)
print("pool with clear hands:", len(pool), dict(reasons))
print("pool by family/style:", Counter((r["family"], r["style"]) for r in pool))

emb = np.load(ART / "clip_image.npy", mmap_mode="r")
random.seed(42)
strata = defaultdict(list)
for r in pool: strata[(r["family"], r["style"])].append(r)
for v in strata.values():
    random.shuffle(v); v.sort(key=lambda r: -(r["best_hand_conf"] > 0.6))  # prefer confident hands, random otherwise
per_user = Counter(); chosen = []; chosen_vecs = []; drop_ids = defaultdict(set)

def try_add(r):
    if r["id"] in MANUAL: drop_ids["manual_sfw_review"].add(r["id"]); return False
    if r["risk"] > MAX_RISK: drop_ids["clip_risk"].add(r["id"]); return False
    if per_user[r["user"]] >= MAX_PER_USER: drop_ids["author_cap"].add(r["id"]); return False
    v = np.asarray(emb[r["row"]], dtype=np.float32)
    if chosen_vecs and float((np.stack(chosen_vecs) @ v).max()) >= DUP_COS: drop_ids["near_dup"].add(r["id"]); return False
    chosen.append(r); chosen_vecs.append(v); per_user[r["user"]] += 1; return True

for fam, q in QUOTA.items():
    want = {s: q // 3 + (1 if i < q % 3 else 0) for i, s in enumerate(STYLES)}
    got = Counter(); its = {s: iter(strata[(fam, s)]) for s in STYLES}
    for s in STYLES:
        for r in its[s]:
            if got[s] >= want[s]: break
            if try_add(r): got[s] += 1
    short = q - sum(got.values())
    while short > 0:  # fill shortfall from styles that still have images (round-robin)
        progressed = False
        for s in STYLES:
            for r in its[s]:
                if try_add(r): got[s] += 1; short -= 1; progressed = True; break
            if short <= 0: break
        if not progressed: break
    print(fam, dict(got), "short" if short > 0 else "", short if short > 0 else "")

# Top-up to TARGET from whatever strata are least represented (flux/extras run out because of the author cap).
TARGET = sum(QUOTA.values())
iters = {k: iter(v) for k, v in strata.items()}
cnt = Counter((r["family"], r["style"]) for r in chosen); topped = Counter()
chosen_ids = {r["id"] for r in chosen}
while len(chosen) < TARGET and iters:
    k = min(iters, key=lambda k: cnt[k])
    for r in iters[k]:
        if r["id"] in chosen_ids: continue
        if try_add(r): cnt[k] += 1; topped[k] += 1; chosen_ids.add(r["id"]); break
    else:
        del iters[k]
print("top-up:", dict(topped))

with open(ROOT / "data/selected.jsonl", "w") as f:
    for r in chosen: f.write(json.dumps(r) + "\n")
stats = {"criteria": dict(max_clip_risk=MAX_RISK, manual_excluded=len(MANUAL), min_conf=MIN_CONF, min_hand_side_px=MIN_SIDE, max_boxes=MAX_BOXES, max_per_author=MAX_PER_USER,
                          near_dup_cos=DUP_COS, quota=QUOTA),
         "candidates": len(cands), "screen_drop": dict(reasons), "pool": len(pool), "select_drop (unique images skipped while filling quotas)": {k: len(v) for k, v in drop_ids.items()},
         "top_up": {f"{a}/{b}": n for (a, b), n in topped.items()}, "selected": len(chosen), "authors": len(per_user),
         "by_family": dict(Counter(r["family"] for r in chosen)), "by_style": dict(Counter(r["style"] for r in chosen)),
         "by_family_style": {f"{a}/{b}": n for (a, b), n in sorted(Counter((r["family"], r["style"]) for r in chosen).items())},
         "by_baseModel": dict(Counter(r["baseModel"] for r in chosen).most_common())}
json.dump(stats, open(ROOT / "data/selection_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
