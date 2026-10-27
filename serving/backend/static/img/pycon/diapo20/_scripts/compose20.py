"""Diapo 20: renders limpios (banco sintético 0.7731) frente a fotos del examen real (0.2667)."""
import csv, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

REN = Path("base_datos_osea/renders/homo_sapiens")
BENCH = Path("data/real_world_benchmark")
out = Path(sys.argv[1])
BG = (14, 16, 19)
RENDERS = ["femur", "humerus", "cranium", "mandible", "rib", "vertebra", "sacrum", "pelvis"]
PHOTOS = [("hueso_largo", "hueso_largo_humero_01.jpg"), ("cranio", "cranio_07.jpg"), ("mandibula_maxilar", "mandibula_06.jpg"),
          ("vertebra", "vertebra_08.jpg"), ("cranio", "cranio_02.jpg"), ("pelvis", "pelvis_04.jpg"),
          ("vertebra", "vertebra_02.jpg"), ("costilla", "costilla_03.jpg")]

def render_square(bone, size=512):
    """Vista con el hueso más grande (sin tocar borde), recorte cuadrado sobre su propio fondo."""
    best = None
    for png in sorted(REN.glob(f"{bone}/*/*.png")):
        im = Image.open(png).convert("RGB"); a = np.asarray(im).astype(int)
        diff = np.abs(a - a[2, 2]).sum(2) > 40
        ys, xs = np.where(diff)
        if not len(xs) or xs.min() < 3 or ys.min() < 3 or xs.max() > a.shape[1] - 4 or ys.max() > a.shape[0] - 4: continue
        area = (xs.max() - xs.min()) * (ys.max() - ys.min())
        if best is None or area > best[0]: best = (area, png, im, (xs.min(), ys.min(), xs.max(), ys.max()))
    _, png, im, (x0, y0, x1, y1) = best
    side = min(int(max(x1 - x0, y1 - y0) * 1.15), im.width, im.height); cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    l = min(max(0, cx - side // 2), im.width - side); t = min(max(0, cy - side // 2), im.height - side)
    return im.crop((l, t, l + side, t + side)).resize((size, size), Image.LANCZOS), png

def photo_square(path, size=512):
    """Foto entera (sin recortar el hueso) sobre una versión desenfocada de sí misma."""
    im = Image.open(path).convert("RGB")
    bg = im.copy(); s = max(size / bg.width, size / bg.height); bg = bg.resize((int(bg.width * s) + 1, int(bg.height * s) + 1))
    bg = bg.crop(((bg.width - size) // 2, (bg.height - size) // 2, (bg.width - size) // 2 + size, (bg.height - size) // 2 + size))
    bg = Image.blend(bg.filter(ImageFilter.GaussianBlur(24)), Image.new("RGB", (size, size), BG), 0.45)
    fg = im.copy(); fg.thumbnail((size, size), Image.LANCZOS)
    bg.paste(fg, ((size - fg.width) // 2, (size - fg.height) // 2)); return bg

def board(tiles, path, accent):
    """1 grande + 3 pequeñas, como en el boceto."""
    B, S, P = 420, 132, 12
    img = Image.new("RGB", (B + S + 3 * P, B + 2 * P), BG); d = ImageDraw.Draw(img)
    img.paste(tiles[0].resize((B, B)), (P, P)); d.rectangle([P - 1, P - 1, P + B, P + B], outline=accent, width=3)
    for i, t in enumerate(tiles[1:4]):
        x, y = B + 2 * P, P + i * (S + (B - 3 * S) // 2)
        img.paste(t.resize((S, S)), (x, y)); d.rectangle([x - 1, y - 1, x + S, y + S], outline=accent, width=2)
    img.save(path)

(out / "renders").mkdir(parents=True, exist_ok=True); (out / "fotos_reales").mkdir(exist_ok=True)
rtiles, ptiles, rows = [], [], []
for i, bone in enumerate(RENDERS, 1):
    t, png = render_square(bone); t.save(out / "renders" / f"{i:02d}_{bone}.png"); rtiles.append(t)
    rows.append({"fichero": f"renders/{i:02d}_{bone}.png", "origen": str(png), "fuente": "render propio (Blender, fase 1 limpia)"})
manifest = {r["filename"]: r for r in csv.DictReader(open(BENCH / "manifest.csv"))}
for i, (g, fn) in enumerate(PHOTOS, 1):
    t = photo_square(BENCH / g / fn); name = f"{i:02d}_{Path(fn).stem}.jpg"
    t.save(out / "fotos_reales" / name, quality=92); ptiles.append(t)
    rows.append({"fichero": f"fotos_reales/{name}", "origen": str(BENCH / g / fn), "fuente": manifest[fn]["source_url"]})
board(rtiles, out / "01_renders_1+3.png", (64, 200, 180))
board(ptiles, out / "02_fotos_reales_1+3.png", (240, 90, 70))
with open(out / "creditos.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["fichero", "origen", "fuente"]); w.writeheader(); w.writerows(rows)
print("ok")
