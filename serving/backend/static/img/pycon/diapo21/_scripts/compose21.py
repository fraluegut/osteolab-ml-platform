"""Diapo 21: segmentación clásica (Otsu) vs Grounding DINO + SAM, y recortes multi-instancia de costillas."""
import sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, ".")
from shared.cv_extractor import extract as ex

PHOTO = Path("serving/backend/static/img/pycon/femur-real.jpg")
RIBS_SRC = Path("base_datos_osea/Fotos entrenamiento/costilla")
ACC = Path("data/real_photos_accepted")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
CYAN, RED, BG = (40, 220, 200), (235, 80, 60), (14, 16, 19)
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 30)

def overlay(rgb, mask, color, alpha=0.6, outline=True):
    a = rgb.astype(float); m = mask > 0
    a[m] = a[m] * (1 - alpha) + np.array(color) * alpha
    img = a.astype(np.uint8)
    if outline:
        cnts, _ = cv2.findContours((m * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, color, 3)
    return Image.fromarray(img)

def tag(d, xy, text, color):
    tb = d.textbbox(xy, text, font=F); d.rectangle([tb[0] - 8, tb[1] - 6, tb[2] + 8, tb[3] + 6], fill=color)
    d.text(xy, text, font=F, fill=(255, 255, 255))

bgr = cv2.imread(str(PHOTO)); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); h, w = bgr.shape[:2]
Image.fromarray(rgb).save(out / "01_foto_vitrina.jpg", quality=95)

# 1 · Clásico: Otsu puro y el _segment real (Otsu + adaptativo) ----------------
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
_, otsu = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
if cv2.countNonZero(otsu) > otsu.size / 2: otsu = cv2.bitwise_not(otsu)
overlay(rgb, otsu, CYAN, 0.65, outline=False).save(out / "02a_otsu_puro.png")
classic = ex._segment(gray)
overlay(rgb, classic, CYAN, 0.6).save(out / "02b_segmentacion_clasica.png")

# 2 · Grounding DINO -> caja -> SAM -> máscara -------------------------------
pil = Image.fromarray(rgb); dev = "cuda" if torch.cuda.is_available() else "cpu"
proc, model = ex._load_grounding_dino()
inp = proc(images=pil, text=ex.GROUNDED_SAM_TEXT_PROMPT, return_tensors="pt").to(dev)
with torch.no_grad():
    det = proc.post_process_grounded_object_detection(model(**inp), threshold=ex.GROUNDED_SAM_BOX_THRESHOLD,
        text_threshold=ex.GROUNDED_SAM_TEXT_THRESHOLD, target_sizes=[(h, w)])[0]
i = int(det["scores"].argmax()); x0, y0, x1, y1 = map(int, det["boxes"][i].tolist())
boxed = pil.copy(); d = ImageDraw.Draw(boxed)
d.rectangle([x0, y0, x1, y1], outline=RED, width=6); tag(d, (x0 + 4, y0 - 50), f'"bone" {float(det["scores"][i]):.2f}', RED)
boxed.save(out / "03_dino_caja.png")
sam_mask = ex._segment_grounded_sam(bgr)
overlay(rgb, sam_mask, RED, 0.6).save(out / "04_sam_mascara.png")

# 3 · Costillas: foto densa con todas las instancias + recortes aceptados ------
man = pd.read_csv(ACC / "manifest.csv")
c = man[man.filename.str.startswith("costilla_1") & man.notes.fillna("").str.contains("instancia")].copy()
c["src"] = c.notes.str.extract(r"de '([^']+)'")
best_src = c.src.value_counts().idxmax()
src_path = RIBS_SRC / best_src
src_bgr = cv2.imread(str(src_path))
masks = ex._segment_grounded_sam_all(src_bgr)
masks = ex._dedupe_overlapping_masks(masks)
src_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB).astype(float)
rng = np.random.default_rng(3)
for m in masks:
    col = rng.integers(60, 255, 3); src_rgb[m > 0] = src_rgb[m > 0] * 0.45 + col * 0.55
inst = Image.fromarray(src_rgb.astype(np.uint8)); d = ImageDraw.Draw(inst)
for m in masks:
    ys, xs = np.where(m > 0); d.rectangle([xs.min(), ys.min(), xs.max(), ys.max()], outline=(255, 255, 255), width=2)
inst.save(out / "05_costillas_foto_instancias.png")
Image.open(src_path).convert("RGB").save(out / "05_costillas_foto_original.jpg", quality=95)

(out / "costillas").mkdir(exist_ok=True)
crops = []
for fn in sorted(c.filename):
    im = Image.open(ACC / "costilla" / fn).convert("RGB"); im.save(out / "costillas" / fn, quality=95)
    t = Image.new("RGB", (220, 220), (128, 128, 128)); im.thumbnail((220, 220)); t.paste(im, ((220 - im.width) // 2, (220 - im.height) // 2)); crops.append(t)
C = 7; R = (len(crops) + C - 1) // C; P = 10
grid = Image.new("RGB", (C * 220 + (C + 1) * P, R * 220 + (R + 1) * P), BG); d = ImageDraw.Draw(grid)
for k, t in enumerate(crops):
    x, y = P + (k % C) * (220 + P), P + (k // C) * (220 + P); grid.paste(t, (x, y)); d.rectangle([x - 1, y - 1, x + 220, y + 220], outline=(70, 74, 78), width=2)
grid.save(out / "06_costillas_recortes.png")
print(f"otsu_area={cv2.countNonZero(otsu)/otsu.size:.2f} clasica_area={cv2.countNonZero(classic)/classic.size:.2f} sam_area={cv2.countNonZero(sam_mask)/sam_mask.size:.2f}")
print(f"costillas: {len(c)} recortes de {c.src.nunique()} fotos; foto mostrada={best_src} con {len(masks)} instancias detectadas ahora, {int((c.src==best_src).sum())} aceptadas")
