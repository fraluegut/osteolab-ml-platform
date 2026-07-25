"""Compone los renders "ruidosos" (RGBA con sombra de contacto y fondo
transparente — ver `blender/render_bone_noisy.py`) sobre fotos de fondo reales,
y aplica una degradación de calidad barata (reescalado abajo/arriba +
recompresión JPEG) para acercarse a una foto de usuario real en vez de un
recorte compuesto perfecto sobre un fondo nítido.

Espera la misma estructura de directorios que `renders/` (la que consume
`src/cv_extractor/build_dataset.py` en el repo padre):
`<especie>/<hueso>/<especimen>/*_viewNNN.png`. Escribe la misma estructura en
el directorio de salida, mismo nombre de fichero pero en `.jpg` (formato final
tipo foto de usuario, no el PNG con alpha del render intermedio).

El color del hueso, la luz y el fondo real se sortean cada uno por su lado
(ver `render_bone_noisy.py`), así que nada garantiza que el hueso se vea
sobre el fondo elegido — `pick_background_crop` prueba varios candidatos de
fondo/recorte y se queda con uno que tenga contraste de luminancia mínimo
(ver `--min-contrast`), para no generar ejemplos donde el hueso es
prácticamente invisible.

Uso:
    python scripts/compose_noisy.py \\
        --renders-dir renders_noisy --backgrounds-dir data/backgrounds \\
        --out-dir renders_noisy_composited
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

BACKGROUND_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--renders-dir", type=Path, default=Path("renders_noisy"))
    p.add_argument("--backgrounds-dir", type=Path, default=Path("data/backgrounds"))
    p.add_argument("--out-dir", type=Path, default=Path("renders_noisy_composited"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jpeg-quality-min", type=int, default=35)
    p.add_argument("--jpeg-quality-max", type=int, default=85)
    p.add_argument("--downscale-min", type=float, default=0.35,
                    help="fracción mínima de resolución al reducir antes de volver a ampliar")
    p.add_argument("--downscale-max", type=float, default=0.85)
    p.add_argument("--min-contrast", type=float, default=25.0,
                    help="diferencia mínima de luminancia (0-255) entre el hueso y el fondo elegido")
    p.add_argument("--max-bg-tries", type=int, default=6,
                    help="candidatos de fondo/recorte a probar buscando el mínimo de contraste antes de quedarse con el mejor")
    return p.parse_args()


def load_backgrounds(backgrounds_dir):
    paths = sorted(p for p in backgrounds_dir.iterdir() if p.suffix.lower() in BACKGROUND_EXTS)
    if not paths:
        raise SystemExit(f"No hay fotos de fondo en {backgrounds_dir}")
    return paths


def random_background_crop(bg_path, size, rng):
    """Recorte cuadrado aleatorio de un fondo real, escalado al tamaño pedido.

    No basta con redimensionar el fondo entero: eso repetiría el mismo
    encuadre de fondo en cada composición que lo usa. Recortar una región
    aleatoria (45%-100% del lado corto) antes de escalar da variedad real
    entre especímenes distintos que caen sobre el mismo fondo.
    """
    bg = Image.open(bg_path).convert("RGB")
    w, h = bg.size
    side = min(w, h)
    crop_side = max(1, int(side * rng.uniform(0.45, 1.0)))
    x0 = rng.randint(0, w - crop_side) if w > crop_side else 0
    y0 = rng.randint(0, h - crop_side) if h > crop_side else 0
    crop = bg.crop((x0, y0, x0 + crop_side, y0 + crop_side))
    return crop.resize(size, Image.LANCZOS)


def pick_background_crop(render, backgrounds, rng, min_contrast, max_tries):
    """Elige fondo + recorte, evitando que el hueso quede casi invisible.

    El color del hueso, la luz y el fondo real se sortean todos por separado
    (ver `render_bone_noisy.py` y `random_background_crop`) — nada garantiza
    que combinen bien. Comprobado a posteriori sobre 200 renders de muestra:
    ~25% de las composiciones salían con <25 de diferencia de luminancia
    entre hueso y fondo (prácticamente invisible), no es un caso raro de
    "dificultad realista", es señal perdida. Aquí se prueban hasta
    `max_tries` candidatos de fondo/recorte y se usa el primero que supere
    `min_contrast`; si ninguno lo logra, se usa el de mayor contraste visto
    (nunca se descarta el render, solo se elige mejor entre las opciones).
    """
    render_rgb = np.asarray(render.convert("RGB"))
    alpha = np.asarray(render)[..., 3]
    bone_mask = alpha > 200
    if bone_mask.sum() < 20:
        # vista degenerada (hueso casi fuera de cuadro) — no hay con qué comparar
        return random_background_crop(rng.choice(backgrounds), render.size, rng)

    bone_lum = render_rgb[bone_mask].mean()
    best_crop, best_diff = None, -1.0
    for _ in range(max_tries):
        crop = random_background_crop(rng.choice(backgrounds), render.size, rng)
        bg_lum = np.asarray(crop)[~bone_mask].mean()
        diff = abs(bone_lum - bg_lum)
        if diff > best_diff:
            best_crop, best_diff = crop, diff
        if diff >= min_contrast:
            return crop
    return best_crop


def degrade_resolution(img, rng, downscale_min, downscale_max):
    # Reducir y volver a ampliar simula la borrosidad de una cámara de móvil
    # mediocre o una foto redimensionada varias veces — barato en Python,
    # evita tener que generar este tipo de ruido dentro de Blender.
    w, h = img.size
    factor = rng.uniform(downscale_min, downscale_max)
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def compose_one(render_path, backgrounds, rng, args):
    render = Image.open(render_path).convert("RGBA")
    bg_crop = pick_background_crop(render, backgrounds, rng, args.min_contrast, args.max_bg_tries)
    composed = Image.alpha_composite(bg_crop.convert("RGBA"), render).convert("RGB")
    return degrade_resolution(composed, rng, args.downscale_min, args.downscale_max)


def main():
    args = parse_args()
    if not args.renders_dir.is_dir():
        raise SystemExit(f"No existe el directorio de renders: {args.renders_dir}")
    backgrounds = load_backgrounds(args.backgrounds_dir)

    render_paths = sorted(args.renders_dir.glob("*/*/*/*.png"))
    if not render_paths:
        raise SystemExit(
            f"No se encontraron renders bajo {args.renders_dir} "
            "(¿estructura especie/hueso/especimen/*.png?)"
        )

    rng = random.Random(args.seed)
    for i, render_path in enumerate(render_paths):
        rel = render_path.relative_to(args.renders_dir)
        out_path = (args.out_dir / rel).with_suffix(".jpg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        composed = compose_one(render_path, backgrounds, rng, args)
        quality = rng.randint(args.jpeg_quality_min, args.jpeg_quality_max)
        composed.save(out_path, format="JPEG", quality=quality)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(render_paths)}")

    print(f"COMPOSE_DONE renders={len(render_paths)} out={args.out_dir}")


if __name__ == "__main__":
    main()
