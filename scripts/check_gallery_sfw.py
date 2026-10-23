"""SFW check for the documentation gallery: the same rule as the external-data prep (Falconsai nsfw_image_detection
P(nsfw) >= 0.2, or CLIP-B/32 laion2b zero-shot 'suggestive' share >= 0.5) applied to each crop and to its source image.
Usage: python scripts/check_gallery_sfw.py ITEM [ITEM ...]
ITEM is a gold key like 129846139_2 (reads the local, unpublished gold crops and images) or a crop path; for a path,
the source image is expected next to it as <name>__full.jpg."""
import json, os, sys, torch, open_clip
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
CROPS = 'external_experiments/kaggle/inputs/gold_crops'; IMGS = 'autolabel_data/hadm/input_all/images'
SUG = ['a photo of a nude person', 'a photo of a person in underwear', 'a photo of a person in lingerie', 'a photo of a person in a bikini',
       'a photo of a person in a swimsuit', 'a sexy pin-up picture', 'an erotic picture', 'a shirtless person', 'a topless person']
SAFE = ['a photo of a fully clothed person', 'a photo of people in everyday clothes', 'an illustration of a person', 'a photo of a landscape',
        'a photo of an object', 'a cartoon character', 'a photo of a person in a coat', 'a photo of a person in a uniform']
proc = AutoImageProcessor.from_pretrained('Falconsai/nsfw_image_detection')
nsfw = AutoModelForImageClassification.from_pretrained('Falconsai/nsfw_image_detection').eval()
NID = [i for i, l in nsfw.config.id2label.items() if l.lower() == 'nsfw'][0]
cb, _, pre = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k'); cb.eval(); tok = open_clip.get_tokenizer('ViT-B-32')
with torch.no_grad():
    TE = cb.encode_text(tok(SUG + SAFE)); TE = TE / TE.norm(dim=-1, keepdim=True)
def check(path):
    im = Image.open(path).convert('RGB')
    with torch.no_grad():
        pn = torch.softmax(nsfw(**proc(images=im, return_tensors='pt')).logits, -1)[0, NID].item()
        f = cb.encode_image(pre(im)[None]); f = f / f.norm(dim=-1, keepdim=True)
        ps = torch.softmax(100 * f @ TE.T, -1)[0, :len(SUG)].sum().item()
    return dict(p_nsfw=round(pn, 4), p_suggestive=round(ps, 4), sfw=pn < 0.2 and ps < 0.5)
out = {}
for k in sys.argv[1:]:
    if '/' in k: out[k] = dict(crop=check(k), image=check(k[:-4] + '__full.jpg'))
    else: out[k] = dict(crop=check(f'{CROPS}/{k}_p15.jpg'), image=check(f'{IMGS}/{k.rsplit("_", 1)[0]}.jpg'))
    print(k, out[k], flush=True)
print('ALL SFW' if all(v['crop']['sfw'] and v['image']['sfw'] for v in out.values()) else 'SOME FLAGGED')
json.dump(out, open(os.environ.get('SFW_OUT', 'readme_preview/gallery_sfw_check.json'), 'w'), indent=1)
