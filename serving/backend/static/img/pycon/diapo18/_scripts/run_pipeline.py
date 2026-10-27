"""Pasa una foto real por el pipeline de OsteoLab y guarda las salidas crudas."""
import base64, io, json, sys
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image

sys.path.insert(0, ".")
from shared.cv_extractor import extract as ex
from shared.clip_filter.predict import classify_image as clip_classify
from shared.inference.predict_two_stage import classify_image_two_stage

photo, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

res = {"clip": clip_classify(open(photo, "rb"))}

# Caja de Grounding DINO (la misma llamada que _segment_grounded_sam)
bgr = cv2.imread(str(photo)); h, w = bgr.shape[:2]
pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
proc, model = ex._load_grounding_dino()
dev = "cuda" if torch.cuda.is_available() else "cpu"
inp = proc(images=pil, text=ex.GROUNDED_SAM_TEXT_PROMPT, return_tensors="pt").to(dev)
with torch.no_grad():
    det = proc.post_process_grounded_object_detection(model(**inp), threshold=ex.GROUNDED_SAM_BOX_THRESHOLD,
        text_threshold=ex.GROUNDED_SAM_TEXT_THRESHOLD, target_sizes=[(h, w)])[0]
i = int(det["scores"].argmax())
res["dino_box"] = [round(v, 1) for v in det["boxes"][i].tolist()]
res["dino_score"] = round(float(det["scores"][i]), 4)
mask = ex._segment_grounded_sam(bgr)
cv2.imwrite(str(out / "mask.png"), mask)

feats = ex.extract_features(open(photo, "rb"), backend="grounded_sam", include_crop=True)
crop = feats.pop("crop_image"); crop.save(out / "crop.png")
for k in ("annotated_image_base64", "mask_overlay_base64"):
    if feats.get(k):
        (out / f"{k.split('_base64')[0]}.png").write_bytes(base64.b64decode(feats.pop(k)))
res["features"] = feats

with open(photo, "rb") as f:
    two = classify_image_two_stage(f)
two.pop("cv_features", None)
res["two_stage"] = two
(out / "pipeline.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
print(json.dumps({k: res[k] for k in ("clip", "dino_box", "dino_score", "two_stage")}, indent=1, ensure_ascii=False, default=str))
