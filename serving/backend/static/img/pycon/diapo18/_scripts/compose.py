"""Compone las imágenes de la diapo 18 a partir de las salidas reales del pipeline."""
import json, sys
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image, ImageDraw, ImageFont

photo, render, raw, out = map(Path, sys.argv[1:5])
out.mkdir(parents=True, exist_ok=True)
P = json.loads((raw / "pipeline.json").read_text())
BG, CARD, LINE, INK, DIM, TEAL, GREEN, BLUE = (14, 16, 19), (24, 27, 31), (58, 64, 70), (236, 236, 232), (150, 156, 160), (64, 200, 180), (90, 210, 120), (52, 152, 235)
FD = "/usr/share/fonts/truetype/dejavu/"
def font(sz, bold=False, mono=True):
    name = ("DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf") if mono else ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    try: return ImageFont.truetype(FD + name, sz)
    except OSError: return ImageFont.truetype(matplotlib.get_data_path() + "/fonts/ttf/" + name, sz)

def canvas(w, h): return Image.new("RGB", (w, h), BG)
def fit(img, w, h, bg=BG):
    img = img.copy(); img.thumbnail((w, h), Image.LANCZOS)
    c = Image.new("RGB", (w, h), bg); c.paste(img, ((w - img.width) // 2, (h - img.height) // 2)); return c
def cover(img, w, h, cx=None, cy=None):
    """Recorta al aspecto w:h centrado en (cx, cy) y escala: sin franjas."""
    cx = img.width / 2 if cx is None else cx; cy = img.height / 2 if cy is None else cy
    r = w / h
    cw, ch = (img.width, img.width / r) if img.width / img.height < r else (img.height * r, img.height)
    l = min(max(0, cx - cw / 2), img.width - cw); t = min(max(0, cy - ch / 2), img.height - ch)
    return img.crop((int(l), int(t), int(l + cw), int(t + ch))).resize((w, h), Image.LANCZOS)
def framed(base, img, xy, color=LINE, width=3):
    base.paste(img, xy); d = ImageDraw.Draw(base)
    d.rectangle([xy[0] - 1, xy[1] - 1, xy[0] + img.width, xy[1] + img.height], outline=color, width=width)
def arrow(d, x0, y, x1, color=INK, w=4):
    d.line([(x0, y), (x1, y)], fill=color, width=w); d.polygon([(x1, y), (x1 - 16, y - 10), (x1 - 16, y + 10)], fill=color)
def table(d, x, y, rows, w, rh=44, hl=0, fs=22):
    for i, (k, v) in enumerate(rows):
        top = y + i * rh
        d.rectangle([x, top, x + w, top + rh - 4], fill=(30, 80, 130) if i == hl else CARD)
        d.text((x + 14, top + rh / 2 - 2), k, font=font(fs, i == hl), fill=INK, anchor="lm")
        d.text((x + w - 14, top + rh / 2 - 2), v, font=font(fs, i == hl), fill=INK, anchor="rm")

ph = Image.open(photo).convert("RGB")
mask = cv2.imread(str(raw / "mask.png"), 0)
x0, y0, x1, y1 = map(int, P["dino_box"])
pad = 40
bone_view = ph.crop((max(0, x0 - pad), max(0, y0 - pad * 2), min(ph.width, x1 + pad), min(ph.height, y1 + pad * 2)))

# 1 · Malla → imagen ------------------------------------------------------------
m = json.loads((raw / "mesh.json").read_text())
V = np.array(m["v"]); F = m["f"]
c = V - V.mean(0); _, _, vt = np.linalg.svd(c, full_matrices=False); pr = c @ vt.T
xy = np.c_[-pr[:, 1], pr[:, 0]]
a = np.deg2rad(22); xy = xy @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
edges = {tuple(sorted((f[i], f[(i + 1) % len(f)]))) for f in F for i in range(len(f))}
segs = [(xy[i], xy[j]) for i, j in edges]
fig = plt.figure(figsize=(5, 5), dpi=120, facecolor=np.array(BG) / 255)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(np.array(BG) / 255)
ax.add_collection(LineCollection(segs, colors=[(0.86, 0.88, 0.9, 0.55)], linewidths=0.35))
ax.set_aspect("equal"); ax.autoscale(); ax.margins(0.06); ax.axis("off")
fig.savefig(out / "_wire.png", facecolor=fig.get_facecolor()); plt.close(fig)
wire = Image.open(out / "_wire.png").convert("RGB"); (out / "_wire.png").unlink()
img = canvas(1000, 480); d = ImageDraw.Draw(img)
framed(img, fit(wire, 400, 400), (40, 20))
arrow(d, 465, 240, 535)
framed(img, fit(Image.open(render).convert("RGB"), 400, 400), (560, 20))
d.text((240, 458), f"malla 3D · {m['orig_faces']:,} caras".replace(",", "."), font=font(18), fill=DIM, anchor="ms")
d.text((760, 458), "render de laboratorio (Blender)", font=font(18), fill=DIM, anchor="ms")
img.save(out / "01_malla_a_imagen.png")

# 2 · ¿Es un hueso? (CLIP) --------------------------------------------------------
clip = P["clip"]["probabilities"]
img = canvas(1000, 480); d = ImageDraw.Draw(img)
framed(img, cover(ph, 560, 380, (x0 + x1) / 2, (y0 + y1) / 2), (20, 50), TEAL)
table(d, 610, 150, [("hueso", f"{clip['bone']:.2f}"), ("no hueso", f"{clip['not_bone']:.2f}")], 370, rh=62, fs=28)
d.text((610, 120), "CLIP zero-shot", font=font(20), fill=DIM)
d.text((610, 290), "foto real · fémur en vitrina", font=font(18), fill=DIM)
img.save(out / "02_es_un_hueso_clip.png")

# 3 · Recortar (Grounding DINO + SAM) ---------------------------------------------
arr = np.array(ph).astype(float)
sel = mask > 0
arr[sel] = arr[sel] * 0.45 + np.array(GREEN) * 0.55
ov = Image.fromarray(arr.astype(np.uint8)); d2 = ImageDraw.Draw(ov)
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cn in cnts:
    if cv2.contourArea(cn) > 200: d2.line([tuple(p[0]) for p in cn] + [tuple(cn[0][0])], fill=GREEN, width=3)
d2.rectangle([x0, y0, x1, y1], outline=TEAL, width=5)
lbl = f"bone {P['dino_score']:.2f}"
tb = d2.textbbox((x0, y0 - 34), lbl, font=font(26, True))
d2.rectangle([tb[0] - 6, tb[1] - 4, tb[2] + 6, tb[3] + 4], fill=TEAL); d2.text((x0, y0 - 34), lbl, font=font(26, True), fill=BG)
img = canvas(1000, 480); framed(img, cover(ov, 960, 420, (x0 + x1) / 2, (y0 + y1) / 2), (20, 14), TEAL)
ImageDraw.Draw(img).text((500, 466), "caja: Grounding DINO («bone.») · máscara: SAM (mayor área)", font=font(17), fill=DIM, anchor="ms")
img.save(out / "03_recortar_dino_sam.png")

# 4 · Medir forma (OpenCV) --------------------------------------------------------
f = P["features"]
ys, xs = np.where(sel); bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
cut = np.array(ph)[by0 - 10:by1 + 10, bx0 - 10:bx1 + 10].copy()
msk = mask[by0 - 10:by1 + 10, bx0 - 10:bx1 + 10] > 0
cut[~msk] = BG
bone = Image.fromarray(cut).rotate(90, expand=True, fillcolor=BG)
img = canvas(1000, 480); d = ImageDraw.Draw(img)
bfit = fit(bone, 300, 400); framed(img, bfit, (40, 30))
bb = bfit.getbbox()
d.line([(370, 50), (370, 410)], fill=INK, width=3)
for yy, s in ((50, 1), (410, -1)): d.polygon([(370, yy), (362, yy + 14 * s), (378, yy + 14 * s)], fill=INK)
d.text((390, 230), "eje", font=font(18), fill=DIM, anchor="lm")
rows = [("aspect_ratio", f"{f['aspect_ratio']:.2f}"), ("elongation", f"{f['elongation']:.2f}"),
        ("solidity", f"{f['solidity']:.2f}"), ("circularity", f"{f['circularity']:.2f}"),
        ("tortuosity", f"{f['tortuosity']:.2f}"), ("n_holes", f"{f['n_holes']}")]
table(d, 470, 50, rows, 500, rh=60, hl=-1, fs=26)
d.text((470, 440), "medidas reales de esta foto · extract.py", font=font(17), fill=DIM)
img.save(out / "04_medir_forma.png")

# 5 · Ver textura -----------------------------------------------------------------
cx = int(x0 + (x1 - x0) * 0.55); cy = int(np.median(np.where(mask[:, cx] > 0)[0]))
s = 44; zoom = ph.crop((cx - s, cy - s, cx + s, cy + s)).resize((360, 360), Image.LANCZOS)
img = canvas(1000, 480); d = ImageDraw.Draw(img)
VW, VH = 540, 400; bcx, bcy = (x0 + x1) / 2, (y0 + y1) / 2
rr = VW / VH; ch = min(ph.height, (x1 - x0 + 2 * pad) / rr); cw = ch * rr
L = min(max(0, bcx - cw / 2), ph.width - cw); T = min(max(0, bcy - ch / 2), ph.height - ch)
framed(img, ph.crop((int(L), int(T), int(L + cw), int(T + ch))).resize((VW, VH), Image.LANCZOS), (20, 40))
sc = VW / cw; ox, oy = 20 - L * sc, 40 - T * sc
r = [ox + (cx - s) * sc, oy + (cy - s) * sc, ox + (cx + s) * sc, oy + (cy + s) * sc]
d.rectangle(r, outline=INK, width=3)
framed(img, zoom, (620, 60), INK)
d.line([(r[2], r[1]), (620, 60)], fill=INK, width=2); d.line([(r[2], r[3]), (620, 420)], fill=INK, width=2)
d.text((800, 450), "detalle ×4: lo que ve la ResNet18", font=font(17), fill=DIM, anchor="ms")
img.save(out / "05_ver_textura.png")

# 6 · Decidir (híbrido + etapa 2) -------------------------------------------------
ts = P["two_stage"]; probs = sorted(ts["probabilities"].items(), key=lambda kv: -kv[1])
NAMES = {"hueso_largo": "Hueso largo", "costilla": "Costilla", "hueso_plano": "Hueso plano", "hueso_pequeno": "Hueso pequeño",
         "vertebra": "Vértebra", "mandibula_maxilar": "Mandíbula", "cranio": "Cráneo", "pelvis": "Pelvis", "sacro": "Sacro",
         "femur": "Fémur", "humero": "Húmero", "hueso_largo_otro": "Otro largo"}
top = probs[:4] + [("otros", sum(p for _, p in probs[4:]))]
img = canvas(1000, 480); d = ImageDraw.Draw(img)
d.text((20, 18), "etapa 1 · grupo", font=font(18), fill=DIM)
for i, (k, p) in enumerate(top):
    y = 60 + i * 50
    d.text((20, y + 14), NAMES.get(k, "Otros"), font=font(22, i == 0), fill=INK, anchor="lm")
    d.rectangle([230, y, 480, y + 28], fill=CARD); d.rectangle([230, y, 230 + 250 * p, y + 28], fill=BLUE if i == 0 else (110, 116, 122))
    d.text((560, y + 14), f"{p:.2f}", font=font(22, i == 0), fill=INK, anchor="rm")
fb = sorted(ts["fine_bone"]["probabilities"].items(), key=lambda kv: -kv[1])
d.text((20, 330), "etapa 2 · hueso concreto", font=font(18), fill=DIM)
for i, (k, p) in enumerate(fb):
    y = 365 + i * 36
    d.text((20, y + 11), NAMES[k], font=font(19, i == 0), fill=INK, anchor="lm")
    d.rectangle([230, y, 480, y + 22], fill=CARD); d.rectangle([230, y, 230 + 250 * p, y + 22], fill=TEAL if i == 0 else (110, 116, 122))
    d.text((560, y + 11), f"{p:.2f}", font=font(19, i == 0), fill=INK, anchor="rm")
crop = Image.open(raw / "crop.png").convert("RGB").rotate(90, expand=True, fillcolor=(128, 128, 128))
cf = fit(crop, 320, 340, (128, 128, 128)); framed(img, cf, (640, 20), TEAL)
d.rectangle([640, 370, 960, 440], fill=CARD, outline=TEAL, width=3)
d.text((800, 405), f"Fémur · {fb[0][1]:.2f}", font=font(30, True), fill=INK, anchor="mm")
img.save(out / "06_decidir.png")

# Piezas sueltas por si las quieres montar a mano
ph.save(out / "foto_original.jpg", quality=95); ov.save(out / "pieza_mascara_sam.png")
zoom.save(out / "pieza_detalle_textura.png"); bone.save(out / "pieza_hueso_recortado.png"); wire.save(out / "pieza_malla_wireframe.png")
print("ok")
