# Hand-defect labeling package

This package contains 2,500 AI-generated images (from Civitai, strict SFW, 512 px wide) with **5,604 pre-annotated hand boxes**,
all labelled `hand_ok`. Your job is to review and fix the boxes and mark broken hands as `hand_bad`.
Read **LABELING_GUIDE.md** first, which takes 5 minutes.

```
hand_defects_labeling/
  images/                         2,500 .webp images (file name = Civitai image id)
  LABELING_GUIDE.md               class definitions, defect rules, edge cases
  docs/                           example grids used by the guide
  labelstudio/
    label_config.xml              labeling interface (paste into Label Studio)
    tasks_local_files.json        tasks + pre-annotations (Option A, recommended)
    tasks_http_server.json        same tasks, images served by a tiny local web server (Option B)
    consistency_recheck_tasks.json  120 tasks WITHOUT pre-boxes, to relabel ~1 week later
  coco/
    annotations/instances_default.json   COCO pre-annotations
    cvat_coco_import.zip                 the same, zipped for CVAT "COCO 1.0" import
  yolo/                           Ultralytics YOLO labels (class 0 = hand_ok, 1 = hand_bad) + data.yaml
  meta/                           per-image metadata (base model, style, author, Civitai page), selection + benchmark stats
```

## Option A: Label Studio with local files (recommended)

Works on Windows, macOS and Linux. You need Python 3.10–3.12 and about 1 GB of free disk space.

1. **Unzip** `hand_defects_labeling.zip`, for example to `C:\data\hand_defects_labeling` (Windows) or
   `~/data/hand_defects_labeling` (macOS). Below, this folder is called `<PKG>`. Use a path **without spaces or Cyrillic letters**.
2. **Install Label Studio** in its own virtual environment:
   ```bash
   python -m venv ls-venv
   # Windows:  ls-venv\Scripts\activate        macOS/Linux:  source ls-venv/bin/activate
   pip install label-studio
   ```
3. **Allow local file serving and start it.** The document root must be `<PKG>`:
   - Windows (PowerShell):
     ```powershell
     $env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"
     $env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="C:\data\hand_defects_labeling"
     label-studio start
     ```
   - macOS/Linux:
     ```bash
     export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
     export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=~/data/hand_defects_labeling
     label-studio start
     ```
   The browser opens http://localhost:8080. Create a local account; it stays on your PC.
4. **Create a project** called `hand-defects`. Under **Labeling Setup**, choose **Custom template**, open the **Code** tab, replace
   everything with the contents of `labelstudio/label_config.xml`, and click **Save**.
5. **Connect the image folder:** go to Project **Settings → Cloud Storage → Add Source Storage → Local files**.
   Set the absolute local path to `<PKG>\images` (Windows) or `<PKG>/images` (macOS), then click **Check connection** and **Add storage**.
   **Do not press "Sync"**: the tasks come from the JSON file in the next step.
6. **Import tasks:** click **Import**, upload `labelstudio/tasks_local_files.json`, and confirm. You should see 2,500 tasks, each with a
   prediction (the pre-boxes). If the images do not load, re-check steps 3 and 5. The usual cause is a different `DOCUMENT_ROOT` path.
7. **Label.** Click **Label All Tasks**. The pre-boxes appear as editable regions. Fix them following the guide, then click **Submit**
   (`Ctrl+Enter`). Progress is saved in Label Studio's database, so you can stop at any time.
   - Optional: in **Settings → Annotation**, enable "Use predictions to prelabel tasks" if the pre-boxes do not appear automatically.
8. **Export** when you are done, or after each batch: **Export → JSON** (the full export, which keeps defect types and skip flags) **and**
   **Export → YOLO** (boxes only). Send both files back. They go to `data/labels/` in the project repository.

## Option B: Label Studio with a tiny local web server

Use this if Option A's local storage gives you trouble. It needs no environment variables and no storage settings.

1. In a separate terminal: `cd <PKG>` and run `python -m http.server 8081 --bind 127.0.0.1`. Keep it running while you label.
2. Run `label-studio start`, create the project, and paste `label_config.xml` (step 4 above).
3. Import `labelstudio/tasks_http_server.json`. The images are loaded from `http://localhost:8081/images/...`.

## Option C: CVAT

1. Create a task with labels `hand_ok` and `hand_bad` (rectangle), and upload all files from `images/`.
2. In the task, choose **Actions → Upload annotations → COCO 1.0** and select `coco/cvat_coco_import.zip`. The boxes are matched by file name.
3. Label, then export as **COCO 1.0** or **Ultralytics YOLO Detection**. Defect types are not part of this setup. Use Label Studio if you want them.

## Consistency re-check (after about 1 week)

Create a **second** project with the same label config and import `labelstudio/consistency_recheck_tasks.json` (120 images,
no pre-boxes, chosen at random). Label them from scratch and export JSON. The script compares them with your first pass.
