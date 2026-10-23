"""Stage 3: build the labeling package (Label Studio + CVAT/COCO + Ultralytics YOLO) from preannotations.jsonl.

Output: dist/hand_defects_labeling/ and dist/hand_defects_labeling.zip
"""
import json, shutil, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent; SRC = Path("/workspace/rag_civitai")
PKG = ROOT / "dist/hand_defects_labeling"
CLASSES = ["hand_ok", "hand_bad"]
rows = [json.loads(l) for l in open(ROOT / "data/preannotations.jsonl")]
rows.sort(key=lambda r: r["id"])
if PKG.exists(): shutil.rmtree(PKG)
for d in ["images", "labelstudio", "yolo/labels/train", "coco/annotations", "meta"]:
    (PKG / d).mkdir(parents=True)

LS_CONFIG = """<View>
  <Header size="5" value="Box every hand (longest side 24 px or more). 1 = hand_ok, 2 = hand_bad. Rules: LABELING_GUIDE.md"/>
  <Text name="info" value="$info"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image" strokeWidth="2">
    <Label value="hand_ok" background="#22c55e" hotkey="1"/>
    <Label value="hand_bad" background="#ef4444" hotkey="2"/>
  </RectangleLabels>
  <Choices name="defect_type" toName="image" perRegion="true" choice="multiple" showInline="true"
           visibleWhen="region-selected" whenTagName="label" whenLabelValue="hand_bad">
    <Choice value="extra_fingers"/><Choice value="missing_fingers"/><Choice value="fused_fingers"/>
    <Choice value="broken_joint_or_wrist"/><Choice value="mangled_blob"/><Choice value="other"/>
  </Choices>
  <Choices name="image_flag" toName="image" choice="multiple" showInline="true">
    <Choice value="skip_unclear"/><Choice value="skip_unsafe"/>
  </Choices>
</View>
"""
(PKG / "labelstudio/label_config.xml").write_text(LS_CONFIG)

tasks_lf, tasks_http, coco_imgs, coco_anns = [], [], [], []
aid = 1
for i, r in enumerate(rows, 1):
    src = SRC / r["path"]; fn = f"{r['id']}.webp"
    shutil.copy2(src, PKG / "images" / fn)
    W, H = r["W"], r["H"]
    hints = [b.get("defect_hint", 0) for b in r["boxes"]]
    results = []
    for j, b in enumerate(r["boxes"]):
        x0, y0, x1, y1 = b["xyxy"]
        results.append({"id": f"r{r['id']}_{j}", "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                        "original_width": W, "original_height": H, "image_rotation": 0, "score": b["score"],
                        "value": {"x": 100 * x0 / W, "y": 100 * y0 / H, "width": 100 * (x1 - x0) / W,
                                  "height": 100 * (y1 - y0) / H, "rotation": 0, "rectanglelabels": ["hand_ok"]}})
        coco_anns.append({"id": aid, "image_id": i, "category_id": 1, "bbox": [x0, y0, x1 - x0, y1 - y0],
                          "area": (x1 - x0) * (y1 - y0), "iscrowd": 0,
                          "attributes": {"det_score": b["score"], "defect_hint": b.get("defect_hint")}})
        aid += 1
    info = (f"id {r['id']} | {r['family']} / {r['baseModel']} | style {r['style']} | "
            f"pre-boxes {len(r['boxes'])} | defect hint max {max(hints, default=0):.2f} (weak CLIP hint, not a label)")
    common = {"civitai_id": r["id"], "family": r["family"], "base_model": r["baseModel"], "style": r["style"],
              "author": r["user"], "page_url": r["page_url"], "n_prebox": len(r["boxes"]),
              "defect_hint_max": round(max(hints, default=0), 3),
              "defect_hints": ", ".join(f"{h:.2f}" for h in hints), "info": info}
    pred = [{"model_version": "adetailer_v8s+v9c_nms", "score": round(max([b["score"] for b in r["boxes"]], default=0), 3),
             "result": results}]
    tasks_lf.append({"data": {"image": f"/data/local-files/?d=images/{fn}", **common}, "predictions": pred})
    tasks_http.append({"data": {"image": f"http://localhost:8081/images/{fn}", **common}, "predictions": pred})
    coco_imgs.append({"id": i, "file_name": fn, "width": W, "height": H})
    with open(PKG / "yolo/labels/train" / f"{r['id']}.txt", "w") as f:
        for b in r["boxes"]:
            x0, y0, x1, y1 = b["xyxy"]
            f.write(f"0 {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} {(x1-x0)/W:.6f} {(y1-y0)/H:.6f}\n")

json.dump(tasks_lf, open(PKG / "labelstudio/tasks_local_files.json", "w"))
import random
random.seed(123)
recheck = [{"data": t["data"]} for t in random.sample(tasks_lf, 120)]  # no predictions: label from scratch
json.dump(recheck, open(PKG / "labelstudio/consistency_recheck_tasks.json", "w"))
(ROOT / "data/consistency_recheck_ids.txt").write_text("".join(f"{t['data']['civitai_id']}\n" for t in recheck))
json.dump(tasks_http, open(PKG / "labelstudio/tasks_http_server.json", "w"))
coco = {"info": {"description": "AI-generated hand defects - pre-annotations (all boxes hand_ok; to be reviewed)"},
        "licenses": [], "categories": [{"id": 1, "name": "hand_ok", "supercategory": ""},
                                       {"id": 2, "name": "hand_bad", "supercategory": ""}],
        "images": coco_imgs, "annotations": coco_anns}
json.dump(coco, open(PKG / "coco/annotations/instances_default.json", "w"))
# CVAT "COCO 1.0" import expects a zip with annotations/instances_default.json
with zipfile.ZipFile(PKG / "coco/cvat_coco_import.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(PKG / "coco/annotations/instances_default.json", "annotations/instances_default.json")
(PKG / "yolo/data.yaml").write_text("path: .\ntrain: images/train\nval: images/train\nnames:\n  0: hand_ok\n  1: hand_bad\n")
(PKG / "yolo/train.txt").write_text("".join(f"images/train/{r['id']}.webp\n" for r in rows))
# CVAT "Ultralytics YOLO Detection 1.0" import zip (labels only; images are uploaded when creating the task)
with zipfile.ZipFile(PKG / "yolo/cvat_yolo_import.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(PKG / "yolo/data.yaml", "data.yaml"); z.write(PKG / "yolo/train.txt", "train.txt")
    for r in rows:
        z.write(PKG / "yolo/labels/train" / f"{r['id']}.txt", f"labels/train/{r['id']}.txt")
with open(PKG / "meta/images.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({k: r[k] for k in ["id", "family", "baseModel", "style", "user", "page_url", "W", "H", "boxes"]}) + "\n")
for p in ["LABELING_GUIDE.md"]:  # lives in docs/; the package keeps its images under docs/
    (PKG / p).write_text((ROOT / "docs" / p).read_text().replace("](examples_", "](docs/examples_"))
shutil.copy2(ROOT / "docs/LABELING_QUICKSTART.md", PKG / "README.md")
(PKG / "docs").mkdir(exist_ok=True)
for p in (ROOT / "docs").glob("*.jpg"): shutil.copy2(p, PKG / "docs" / p.name)
for p in ["selection_stats.json", "bench_scores.json"]:
    shutil.copy2(ROOT / "data" / p, PKG / "meta" / p)
z = ROOT / "dist/hand_defects_labeling.zip"
if z.exists(): z.unlink()
shutil.make_archive(str(z.with_suffix("")), "zip", ROOT / "dist", "hand_defects_labeling")
print("images", len(rows), "boxes", aid - 1, "zip MB", round(z.stat().st_size / 1e6, 1))
