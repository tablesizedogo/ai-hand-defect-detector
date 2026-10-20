"""Stage 1a: gather candidate images that likely show hands, using CLIP text->image retrieval
over the precomputed CLIP ViT-B/32 image embeddings of the Civitai corpus.

Also computes per-image zero-shot style (photo / anime / art) and a "suggestive" score used as an
extra strict-SFW filter on top of the corpus blocklist.
Runs inside the civitai-multimodal-rag search venv (needs open_clip + civitai_search).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SRC = Path("/workspace/rag_civitai")
sys.path.insert(0, str(SRC / "search"))
from civitai_search.encoders import get_clip  # noqa: E402
from civitai_search.safety import blocked_term  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data"
ART = SRC / "search" / "artifacts"

HAND_QUERIES = [
    "a person's hands", "close-up of hands", "hands with visible fingers", "fingers close-up",
    "a hand holding a cup", "holding a cup of coffee with both hands", "holding a mug",
    "a person waving their hand", "raised hand with open palm", "hand making a peace sign",
    "thumbs up gesture", "pointing finger", "hands on face", "hand on cheek", "hand on chin, thinking",
    "praying hands", "hands clasped together", "holding a sword", "holding a gun", "holding a staff",
    "holding a book", "holding a phone", "holding a flower", "holding a sign", "holding an umbrella",
    "holding a glass of wine", "playing the guitar", "playing the piano", "hands typing on a keyboard",
    "holding a paintbrush, painting", "cooking with hands", "holding a fan", "holding a lantern",
    "hands reaching out", "two people holding hands", "handshake", "hand in hair",
    "outstretched arm with open hand", "casting a spell with hands", "hands glowing with magic",
    "fist", "crossed arms", "hand covering mouth", "holding a teacup", "holding a camera",
    "anime girl holding a cup", "anime character waving", "anime character making a peace sign",
    "anime girl with hands together", "anime boy holding a sword", "anime character pointing",
    "illustration of a hand", "a painting of hands", "portrait with hands visible",
    "gloved hands", "hand holding a card", "holding a cigarette", "holding chopsticks, eating",
    "holding a microphone, singing", "holding a bouquet",
]
STYLES = {
    "photo": ["a photograph", "a realistic photo of a person", "a cinematic film still", "a DSLR photo"],
    "anime": ["an anime illustration", "a manga style drawing", "anime girl, cel shading", "a 2D cartoon"],
    "art": ["a digital painting", "an oil painting", "concept art, fantasy illustration", "a 3D render",
            "a watercolor painting"],
}
SUGGESTIVE = ["a photo of a woman in a swimsuit", "lingerie", "a revealing outfit with cleavage",
              "a sexy pin-up pose", "a woman in a tight revealing dress", "bare skin, partially undressed", "a woman with bare shoulders in an off-shoulder dress",
              "a woman in a crop top showing her midriff", "a woman in a short skirt showing legs", "a glamour photoshoot"]
NEUTRAL = ["a fully clothed person", "a person wearing a coat", "a person in casual clothes",
           "a landscape", "a person in armor", "a person in a long dress"]

# second, stricter pass added after visual QA of the first selection (shirtless / blood / revealing outfits)
STRICT_BAD = ["a shirtless man with a bare chest", "a muscular shirtless man", "blood on the face, gore", "a bloody wound",
              "a woman with bare shoulders", "a person in a swimsuit"]
STRICT_NEU = ["a fully clothed person", "a person wearing a shirt", "a person in a coat", "a clean face",
              "a person in armor", "a person in a long dress"]

# "risk" score used by 04_select_final.py (threshold there): fan-service / body-emphasis prompts vs neutral.
RISK_BAD = ["anime girl in a short skirt showing thighs", "a sexy anime girl", "cleavage", "a woman in tight revealing clothes",
            "a woman in shorts showing legs", "a bloody gory scene", "a shirtless man", "a woman in a tank top", "fanservice"]
RISK_NEU = ["a fully clothed person", "an anime girl in a long dress", "a person wearing a sweater", "a person in a coat",
            "a landscape", "an object"]

EXTRA_PROMPT_BLOCK = ["sexy", "seduct", "alluring", "flirt", "pin-up", "pinup", "swimwear", "bathing suit",
                      "corset", "garter", "lace ", "undress", "sultry", "tempting", "bedroom eyes", "kiss",
                      "cosplay girl", "nightgown", "negligee", "body paint", "bodypaint", "nsfw", "suggestive",
                      "busty", "hot girl", "cleav", "bust", "short skirt", "bathing", "beach"]


def main():
    ids = np.array([int(x) for x in open(ART / "clip_image_ids.txt")])
    emb = np.load(ART / "clip_image.npy").astype(np.float32)
    meta = {}
    for line in open(SRC / "meta.jsonl", encoding="utf-8"):
        d = json.loads(line)
        meta[d["id"]] = d
    flags = {}
    for line in open(ART / "clean_flags.jsonl"):
        d = json.loads(line)
        if d.get("text_junk") or d.get("dup_of"):
            flags[d["id"]] = "junk" if d.get("text_junk") else "dup"
    manual = {int(x) for x in open(SRC / "manual_remove.txt").read().split() if x.strip().isdigit()}

    clip = get_clip()
    q = clip.encode_texts(HAND_QUERIES)
    sims = emb @ q.T  # (N, Q)
    hand_score = sims.max(1)
    hand_rank_hits = np.zeros(len(ids), dtype=np.int32)
    TOPK = 900
    cand = set()
    for j in range(len(HAND_QUERIES)):
        top = np.argpartition(-sims[:, j], TOPK)[:TOPK]
        cand.update(top.tolist())
        hand_rank_hits[top] += 1

    style_vecs = {k: clip.encode_texts(v).mean(0) for k, v in STYLES.items()}
    S = np.stack([v / np.linalg.norm(v) for v in style_vecs.values()])
    style_logits = emb @ S.T * 100
    style_idx = style_logits.argmax(1)
    style_names = list(STYLES)
    sug = clip.encode_texts(SUGGESTIVE)
    neu = clip.encode_texts(NEUTRAL)
    sug_score = (emb @ sug.T).max(1) - (emb @ neu.T).max(1)
    risk = (emb @ clip.encode_texts(RISK_BAD).T).max(1) - (emb @ clip.encode_texts(RISK_NEU).T).max(1)
    strict_score = (emb @ clip.encode_texts(STRICT_BAD).T).max(1) - (emb @ clip.encode_texts(STRICT_NEU).T).max(1)

    reasons = Counter()
    rows = []
    for i in sorted(cand):
        iid = int(ids[i])
        m = meta.get(iid)
        if m is None:
            reasons["missing_meta"] += 1; continue
        if iid in flags:
            reasons["clean_flag_" + flags[iid]] += 1; continue
        if iid in manual:
            reasons["manual_remove"] += 1; continue
        p = (m.get("prompt") or "")
        if blocked_term(p):
            reasons["blocklist_en_ru"] += 1; continue
        pl = p.lower()
        if any(t in pl for t in EXTRA_PROMPT_BLOCK):
            reasons["extra_prompt_block"] += 1; continue
        if sug_score[i] > 0.0:
            reasons["clip_suggestive"] += 1; continue
        if strict_score[i] > 0.0:
            reasons["clip_strict_sfw"] += 1; continue
        if not (SRC / m["local_path"]).exists():
            reasons["missing_file"] += 1; continue
        bm = m.get("baseModel") or ""
        fam = ("krea2" if bm.startswith("Krea 2") else "illustrious" if bm == "Illustrious"
               else "flux" if bm.startswith("Flux") else "extras")
        rows.append({"id": iid, "row": int(i), "path": m["local_path"], "family": fam, "baseModel": bm,
                     "user": m.get("username"), "style": style_names[style_idx[i]],
                     "hand_clip": round(float(hand_score[i]), 4), "query_hits": int(hand_rank_hits[i]),
                     "suggestive": round(float(sug_score[i]), 4), "risk": round(float(risk[i]), 4),
                     "width": m.get("width"), "height": m.get("height"),
                     "page_url": m.get("page_url"), "download_url": m.get("download_url")})
    rows.sort(key=lambda r: -r["hand_clip"])
    with open(OUT / "candidates.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats = {"queries": len(HAND_QUERIES), "topk_per_query": TOPK, "union": len(cand),
             "dropped": dict(reasons), "kept": len(rows),
             "by_family": Counter(r["family"] for r in rows), "by_style": Counter(r["style"] for r in rows),
             "corpus_style": Counter(style_names[k] for k in style_idx)}
    json.dump(stats, open(OUT / "candidates_stats.json", "w"), indent=1, default=dict)
    print(json.dumps(stats, indent=1, default=dict))
    # save style for whole corpus row-aligned (handy later)
    np.save(OUT / "row_ids.npy", ids)


if __name__ == "__main__":
    main()
