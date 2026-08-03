"""CLI de `segment_instances()` (ver `extract.py`) — segmenta cada hueso
suelto de una foto con varios (costillas en abanico, vértebras en grid...) y
guarda un recorte por hueso, con el resto pintado de gris. Pensado para
revisar visualmente el resultado ANTES de aceptar nada en
`data/real_photos_accepted/manifest.csv` — el recorte automático puede fallar
si dos huesos están pegados/solapados de verdad.

Uso:
    python -m shared.cv_extractor.segment_instances_cli <imagen_o_carpeta> --out <carpeta_salida>
    python -m shared.cv_extractor.segment_instances_cli <imagen> --prompt "rib." --box-threshold 0.2
"""
import argparse
from pathlib import Path

from shared.cv_extractor.extract import segment_instances


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="imagen suelta o carpeta con varias")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prompt", default="bone.", help="texto para Grounding DINO, ej. 'rib.' o 'bone.'")
    p.add_argument("--box-threshold", type=float, default=0.25)
    return p.parse_args()


def _process_one(img_path: Path, out_dir: Path, prompt: str, box_threshold: float):
    with img_path.open("rb") as fh:
        crops = segment_instances(fh, text_prompt=prompt, box_threshold=box_threshold)
    stem = img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, crop in enumerate(crops, start=1):
        crop.save(out_dir / f"{stem}__instancia_{i:02d}.jpg")
    print(f"{img_path.name}: {len(crops)} instancias -> {out_dir}")
    return len(crops)


def main():
    args = parse_args()
    total = 0
    if args.input.is_dir():
        files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        for f in files:
            total += _process_one(f, args.out, args.prompt, args.box_threshold)
    else:
        total += _process_one(args.input, args.out, args.prompt, args.box_threshold)
    print(f"\nTotal instancias generadas: {total}")


if __name__ == "__main__":
    main()
