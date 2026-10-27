"""Imágenes de la diapo 19 a partir de los renders de laboratorio y del CSV de entrenamiento de hybrid_lab."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

LAB = Path("base_datos_osea/renders_lab")
CSV = Path("data/processed/bone_geometric_features_combined_lab.csv")
D18 = Path("serving/backend/static/img/pycon/diapo18")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
BG, INK, DIM = (14, 16, 19), (236, 236, 232), (150, 156, 160)
FD = "/usr/share/fonts/truetype/dejavu/"
def font(sz, bold=False): return ImageFont.truetype(FD + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), sz)

def best_view(species, bone, size=512, view=None, spec=None):
    """Vista con la máscara más grande que no toca el borde; recorte cuadrado centrado en el hueso."""
    best = None
    dirs = sorted(d for d in (LAB / species / bone).iterdir() if d.is_dir())
    pngs = [q for d in (dirs if spec is None else [dirs[spec]]) for q in sorted(d.glob("*.png"))]
    for png in pngs:
        if png.stem.endswith("_mask"): continue
        if view and view not in png.stem: continue
        m = np.array(Image.open(png.with_name(png.stem + "_mask.png")).convert("L")) > 127
        ys, xs = np.where(m)
        if not len(xs) or (view is None and (xs.min() < 3 or ys.min() < 3 or xs.max() > m.shape[1] - 4 or ys.max() > m.shape[0] - 4)): continue
        if best is None or m.sum() > best[1]: best = (png, m.sum(), (xs.min(), ys.min(), xs.max(), ys.max()))
    png, _, (x0, y0, x1, y1) = best
    im = Image.open(png).convert("RGB")
    side = int(max(x1 - x0, y1 - y0) * 1.18); cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    side = min(side, im.width, im.height)
    l = min(max(0, cx - side // 2), im.width - side); t = min(max(0, cy - side // 2), im.height - side)
    return im.crop((l, t, l + side, t + side)).resize((size, size), Image.LANCZOS), png

src = {}
def save(img, name, png): img.save(out / name); src[name] = str(png)

# 1 · Grupos morfológicos: un render real por grupo -------------------------------
GROUPS = [("cranio", "Cráneo", "homo_sapiens", "cranium"), ("mandibula_maxilar", "Mandíbula / maxilar", "homo_sapiens", "mandible"),
          ("vertebra", "Vértebra", "homo_sapiens", "vertebra", 1, "drop00_view001"), ("sacro", "Sacro", "homo_sapiens", "sacrum"),
          ("costilla", "Costilla", "homo_sapiens", "rib"), ("hueso_largo", "Hueso largo", "homo_sapiens", "femur"),
          ("hueso_plano", "Hueso plano", "homo_sapiens", "scapula"), ("pelvis", "Pelvis", "homo_sapiens", "pelvis"),
          ("hueso_pequeno", "Hueso pequeño", "homo_sapiens", "metatarsal_1", 1, "drop01_view001")]
(out / "grupos").mkdir(exist_ok=True)
tiles = []
for i, (g, label, sp, bone, *pick) in enumerate(GROUPS, 1):
    im, png = best_view(sp, bone, spec=pick[0] if pick else None, view=pick[1] if pick else None); save(im, f"grupos/{i:02d}_{g}.png", png); tiles.append((im, label))
T, PAD, LH = 300, 16, 44
sheet = Image.new("RGB", (3 * T + 4 * PAD, 3 * (T + LH) + 3 * PAD), BG); d = ImageDraw.Draw(sheet)
for i, (im, label) in enumerate(tiles):
    x, y = PAD + (i % 3) * (T + PAD), PAD + (i // 3) * (T + LH + PAD)
    sheet.paste(im.resize((T, T), Image.LANCZOS), (x, y)); d.rectangle([x - 1, y - 1, x + T, y + T], outline=(70, 74, 78), width=2)
    d.text((x + T / 2, y + T + LH / 2), label, font=font(22, True), fill=INK, anchor="mm")
sheet.save(out / "02_nueve_grupos.png")

# 2 · Huesos concretos (fila de abajo) ---------------------------------------------
BONES = [("femur", "Fémur", "gorilla_gorilla"), ("humerus", "Húmero", "homo_sapiens", 0, "drop00_view001"), ("rib", "Costilla", "homo_sapiens"),
         ("tibia", "Tibia", "homo_sapiens"), ("scapula", "Escápula", "homo_sapiens")]
(out / "huesos").mkdir(exist_ok=True)
for i, (bone, label, sp, *pick) in enumerate(BONES, 1):
    im, png = best_view(sp, bone, spec=pick[0] if pick else None, view=pick[1] if pick else None); save(im, f"huesos/{i:02d}_{bone}_{sp}.png", png)

# 3 · Mosaico de renders (variedad de huesos y especies) ---------------------------
MOS = [("canis_familiaris", "cranium"), ("homo_naledi", "clavicle"), ("bos_taurus", "cranium"),
       ("homo_naledi", "radius"), ("homo_sapiens", "metatarsal_1"), ("macaca_radiata", "mandible"),
       ("homo_naledi", "vertebra"), ("homo_sapiens", "ulna"), ("sus_domesticus", "cranium")]
ims = []
for sp, bone in MOS:
    im, png = best_view(sp, bone, 256); ims.append(im); src[f"mosaico:{sp}/{bone}"] = str(png)
M, P = 220, 12
mos = Image.new("RGB", (3 * M + 4 * P, 3 * M + 4 * P), BG); d = ImageDraw.Draw(mos)
for i, im in enumerate(ims):
    x, y = P + (i % 3) * (M + P), P + (i // 3) * (M + P)
    mos.paste(im.resize((M, M), Image.LANCZOS), (x, y)); d.rectangle([x - 1, y - 1, x + M, y + M], outline=(70, 74, 78), width=2)
mos.save(out / "01_mosaico_renders.png")

# 4 · Blender: malla → render (reutiliza la de la diapo 18, mismo fémur de gorila) --
for n in ("pieza_malla_wireframe.png",):
    if (D18 / n).exists(): Image.open(D18 / n).save(out / "00_malla_wireframe.png")
best_view("gorilla_gorilla", "femur")[0].save(out / "00_render_femur_gorila.png")

# 5 · Números reales y gráfico -----------------------------------------------------
df = pd.read_csv(CSV)
real = df["source_path"].astype(str).str.contains("real_photos")
syn = df[~real].assign(k=lambda x: x["species"].astype(str) + "::" + x["specimen"].astype(str))
spec = syn.groupby("bone_group")["k"].nunique().sort_values(ascending=False)
imgs = df.groupby("bone_group").size()
NAMES = {g: l for g, l, *_ in GROUPS}
WRAP = {"hueso_largo": "Hueso\nlargo", "hueso_pequeno": "Hueso\npequeño", "hueso_plano": "Hueso\nplano", "mandibula_maxilar": "Mandíbula/\nmaxilar"}
stats = {
    "fuente": str(CSV) + " (dataset de hybrid_lab, modelo por defecto)",
    "mallas_especimenes_total": int(syn["k"].nunique()), "huesos_distintos": int(syn["bone"].nunique()),
    "especies": int(syn["species"].nunique()), "imagenes_sinteticas": int((~real).sum()), "fotos_reales": int(real.sum()),
    "especimenes_por_grupo": {g: int(v) for g, v in spec.items()},
    "imagenes_por_grupo": {g: int(v) for g, v in imgs.sort_values(ascending=False).items()},
    "desbalance_especimenes": round(float(spec.max() / spec.min()), 1),
    "desbalance_imagenes": round(float(imgs.max() / imgs.min()), 1),
    "imagen_de_cada_fichero": src,
}
(out / "datos_reales.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150, facecolor=np.array(BG) / 255)
ax.set_facecolor(np.array(BG) / 255)
cols = ["#b5651d"] + ["#c89a6a"] * (len(spec) - 2) + ["#8a4b1c"]
bars = ax.bar([WRAP.get(g, NAMES[g]) for g in spec.index], spec.values, color=cols, width=0.68)
for b, v in zip(bars, spec.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.8, str(v), ha="center", va="bottom", color="#ecece8", fontsize=12, fontweight="bold")
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#5a5f64")
ax.tick_params(colors="#c8c8c4", labelsize=10); ax.yaxis.grid(True, color="#2a2e33"); ax.set_axisbelow(True)
ax.set_ylim(0, spec.max() * 1.15)
ax.set_title("Especímenes (mallas) por grupo", color="#ecece8", fontsize=14, fontweight="bold", loc="left")
fig.tight_layout(); fig.savefig(out / "03_mallas_por_grupo.png", facecolor=fig.get_facecolor()); plt.close(fig)
print(json.dumps({k: v for k, v in stats.items() if k != "imagen_de_cada_fichero"}, indent=1, ensure_ascii=False))
