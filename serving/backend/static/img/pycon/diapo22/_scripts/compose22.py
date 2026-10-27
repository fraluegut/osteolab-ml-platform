"""Diapo 22: de la foto a números. Misma geometría que extract.py, dibujada sobre fotos reales del set aceptado."""
import json, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, ".")
from shared.cv_extractor import extract as ex

ACC = Path("data/real_photos_accepted")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
BONES = [("costilla", "Costilla", "costilla/costilla_102.jpg"), ("humero", "Húmero", "hueso_largo/hueso_largo_007.jpg"),
         ("femur", "Fémur", "hueso_largo/hueso_largo_005.jpg"), ("tibia", "Tibia", "hueso_largo/hueso_largo_021.jpg")]
MAIN = "femur"
BG, CARD, INK, DIM, TEAL, BLUE, GOLD = (14, 16, 19), (24, 27, 31), (236, 236, 232), (150, 156, 160), (64, 210, 180), (90, 160, 245), (240, 190, 90)
FD = "/usr/share/fonts/truetype/dejavu/"
def font(sz, bold=False, mono=True):
    return ImageFont.truetype(FD + ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if bold else "") + ".ttf", sz)

def analyse(path):
    """Replica de extract_features: canónica, SAM, contorno, rectángulo rotado y línea central."""
    img = ex._resize_to_canonical(cv2.imread(str(path)))
    binary, contour = ex._locate_bone(img, "grounded_sam")
    feats = ex._compute_all_features(img, binary, contour, include_crop=False)
    for k in ("annotated_image_base64", "mask_overlay_base64"): feats.pop(k, None)
    rect = cv2.minAreaRect(contour); box = cv2.boxPoints(rect).astype(np.float64)
    edges = [np.linalg.norm(box[i] - box[i - 1]) for i in range(4)]; le = int(np.argmax(edges))
    u = box[le] - box[le - 1]; u /= np.linalg.norm(u); v = np.array([-u[1], u[0]]); c = box.mean(0)
    pts = contour.reshape(-1, 2).astype(np.float64); x = (pts - c) @ u; y = (pts - c) @ v
    sxs = np.linspace(x.min(), x.max(), 20); band = (x.max() - x.min()) / 30
    cl, slices = [], []
    for sx in sxs:
        m = np.abs(x - sx) <= band
        if not m.any(): m = np.argsort(np.abs(x - sx))[:5]
        lo, hi = y[m].min(), y[m].max(); cl.append((sx, (lo + hi) / 2)); slices.append((sx, lo, hi))
    ys = np.array([p[1] for p in cl]); sm = np.convolve(ys, np.ones(3) / 3, mode="same"); sm[0], sm[-1] = ys[0], ys[-1]
    to_img = lambda a, b: tuple(c + a * u + b * v)
    return dict(img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), binary=binary, contour=contour, feats=feats, box=box,
                centerline=[to_img(a, b) for a, b in zip(sxs, sm)], slices=[(to_img(a, lo), to_img(a, hi)) for a, lo, hi in slices],
                u=u, v=v, c=c, L=float(x.max() - x.min()), W=float(max(w for _, lo, hi in slices for w in [hi - lo])), rect=rect)

def crop_around(a, pad=0.12, square=False, aspect=None):
    ys, xs = np.where(a["binary"] > 0); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pw, ph = (x1 - x0) * pad + 10, (y1 - y0) * pad + 10
    x0, x1, y0, y1 = x0 - pw, x1 + pw, y0 - ph, y1 + ph
    if square: aspect = 1.0
    if aspect:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; w_, h_ = x1 - x0, y1 - y0
        if w_ / h_ < aspect: w_ = h_ * aspect
        else: h_ = w_ / aspect
        x0, x1, y0, y1 = cx - w_ / 2, cx + w_ / 2, cy - h_ / 2, cy + h_ / 2
    return int(x0), int(y0), int(x1), int(y1)

PADPX = 600
def crop_img(a, b):
    im = a["img"]; edge = np.concatenate([im[0], im[-1], im[:, 0], im[:, -1]])
    fill = [int(x) for x in np.median(edge, axis=0)]
    big = cv2.copyMakeBorder(im, PADPX, PADPX, PADPX, PADPX, cv2.BORDER_CONSTANT, value=fill)
    return Image.fromarray(big[b[1] + PADPX:b[3] + PADPX, b[0] + PADPX:b[2] + PADPX])

def dashed(d, pts, color, width=3, dash=3):
    for i in range(len(pts) - 1):
        if i % 2 == 0: d.line([pts[i], pts[i + 1]], fill=color, width=width)
    for p in pts: d.ellipse([p[0] - dash, p[1] - dash, p[0] + dash, p[1] + dash], fill=color)

def scale_pts(pts, box, s): return [((p[0] - box[0]) * s, (p[1] - box[1]) * s) for p in pts]

R = {k: analyse(ACC / p) for k, _, p in BONES}
a = R[MAIN]; f = a["feats"]

# 1 · Imagen del hueso ---------------------------------------------------------
bx = crop_around(a, pad=0.10, aspect=4 / 3); S = 1200 / (bx[2] - bx[0])
base = crop_img(a, bx).resize((1200, 900), Image.LANCZOS)
base.save(out / "01_imagen_hueso.png")

# 2 · Claves geométricas: contorno, rectángulo rotado, L, W, cortes y línea central
g = base.copy(); d = ImageDraw.Draw(g, "RGBA")
cnt = scale_pts([tuple(p[0]) for p in a["contour"]], bx, S); d.line(cnt + [cnt[0]], fill=(255, 255, 255, 120), width=2)
bpts = scale_pts([tuple(p) for p in a["box"]], bx, S); d.line(bpts + [bpts[0]], fill=GOLD + (200,), width=3)
for lo, hi in a["slices"][1::3]:
    p, q = scale_pts([lo, hi], bx, S); d.line([p, q], fill=BLUE + (170,), width=2)
dashed(d, scale_pts(a["centerline"], bx, S), TEAL, 4, 5)
c, u, v = a["c"], a["u"], a["v"]
half = max(abs((np.asarray(a["box"]) - c) @ v)); side = v if v[1] < 0 else -v; off = side * (half + 22)
p0, p1 = scale_pts([tuple(c - u * a["L"] / 2 + off), tuple(c + u * a["L"] / 2 + off)], bx, S)
d.line([p0, p1], fill=TEAL, width=3)
for p, q in ((p0, p1), (p1, p0)):
    dv = np.subtract(q, p); dv = dv / np.linalg.norm(dv) * 18; n = np.array([-dv[1], dv[0]]) * 0.5
    d.polygon([p, tuple(p + dv + n), tuple(p + dv - n)], fill=TEAL)
mid = np.add(p0, p1) / 2 + side * 34
d.text(tuple(mid), f"L = {a['L']:.0f} px", font=font(30, True), fill=TEAL, anchor="mm")
i_w = int(np.argmax([np.linalg.norm(np.subtract(hi, lo)) for lo, hi in a["slices"]]))
wl, wh = scale_pts(list(a["slices"][i_w]), bx, S); d.line([wl, wh], fill=BLUE, width=5)
dv = np.subtract(wh, wl); dv = dv / np.linalg.norm(dv)
wm = np.add(wh, dv * 40) if wh[1] > wl[1] else np.add(wl, -dv * 40)
d.text(tuple(wm), f"W = {a['W']:.0f} px", font=font(30, True), fill=BLUE, anchor="mm")
g.save(out / "02_claves_geometricas.png")

# 3 · Features numéricas (valores reales de esta foto) ---------------------------
ROWS = [("aspect_ratio (L/W)", "aspect_ratio", 8), ("elongation", "elongation", 1), ("solidity", "solidity", 1),
        ("curvature_mean", "curvature_mean", 0.1), ("tortuosity", "tortuosity", 1.5), ("turning_consistency", "turning_consistency", 1),
        ("n_holes", "n_holes", 5)]
COLS = [(110, 220, 160), (90, 150, 240), (240, 190, 90), (170, 140, 240), (240, 120, 110), (180, 130, 240), (110, 116, 122)]
img = Image.new("RGB", (900, 70 + 64 * len(ROWS)), BG); d = ImageDraw.Draw(img)
for i, ((lbl, k, mx), col) in enumerate(zip(ROWS, COLS)):
    y = 30 + i * 64; val = f[k]
    d.rectangle([20, y, 880, y + 54], fill=CARD); d.text((40, y + 27), lbl, font=font(24), fill=INK, anchor="lm")
    d.rectangle([450, y + 18, 740, y + 36], fill=(40, 44, 50)); d.rectangle([450, y + 18, 450 + 290 * min(1, val / mx), y + 36], fill=col)
    d.text((860, y + 27), f"{val:.2f}" if isinstance(val, float) else str(val), font=font(26, True), fill=col, anchor="rm")
img.save(out / "03_features_numericas.png")

# 4 · Comparativa de 4 huesos ----------------------------------------------------
(out / "huesos").mkdir(exist_ok=True)
cards, stats = [], {}
for k, label, p in BONES:
    r = R[k]; b = crop_around(r, 0.06, square=True); s = 380 / (b[2] - b[0])
    t = crop_img(r, b).resize((380, 380), Image.LANCZOS); dd = ImageDraw.Draw(t)
    dashed(dd, scale_pts(r["centerline"], b, s), TEAL, 3, 4)
    t.save(out / "huesos" / f"{k}.png")
    ff = r["feats"]; stats[k] = {"foto": str(ACC / p), **{n: ff[n] for n in ("aspect_ratio", "elongation", "solidity", "curvature_mean", "tortuosity", "turning_consistency", "total_turning_angle", "n_holes")}}
    card = Image.new("RGB", (400, 600), CARD); cd = ImageDraw.Draw(card)
    cd.text((16, 22), label, font=font(30, True, mono=False), fill=INK, anchor="lm"); card.paste(t, (10, 48))
    for j, (n, key) in enumerate((("L/W", "aspect_ratio"), ("tortuosidad", "tortuosity"), ("consistencia", "turning_consistency"))):
        yy = 455 + j * 46; cd.line([(16, yy - 14), (384, yy - 14)], fill=(50, 54, 60))
        cd.text((20, yy + 6), n, font=font(22), fill=DIM, anchor="lm"); cd.text((380, yy + 6), f"{ff[key]:.2f}", font=font(24, True), fill=INK, anchor="rm")
    cards.append(card)
comp = Image.new("RGB", (4 * 400 + 5 * 16, 632), BG)
for i, cimg in enumerate(cards): comp.paste(cimg, (16 + i * 416, 16))
comp.save(out / "04_comparativa_4_huesos.png")
(out / "datos_reales.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
print(json.dumps({k: {n: v for n, v in s.items() if n != "foto"} for k, s in stats.items()}, indent=1))
