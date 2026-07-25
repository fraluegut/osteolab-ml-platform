"""Fotorrealiza una muestra de renders limpios con Stable Diffusion (img2img,
modelo abierto, se ejecuta en local sin coste) para generar candidatos a
dataset de entrenamiento adicional — la apuesta es cerrar el hueco sim-to-real
sintetizando textura/iluminación realista sobre una forma 3D ya correcta, en
vez de intentar que una GAN aprenda huesos desde cero con las ~75 fotos
reales que tenemos (demasiado pocas para eso, colapsaría).

Importante: esto NO mete nada directamente en el dataset de entrenamiento.
Escribe los candidatos en `data/ai_generated_review/<bone_group>/` +
`manifest.csv` con estado "pending" — hay que revisarlos uno a uno con
`review_ai_generated.py` (Streamlit) antes de que cuenten para nada. Muchas
generaciones de un modelo de imagen genérico (no entrenado en huesos) van a
salir mal (formas imposibles, texturas que no son hueso) — de ahí la revisión
manual en vez de aceptar todo automáticamente.

Uso:
    python -m base_datos_osea.scripts.generate_ai_variants --n-per-group 4
    python -m base_datos_osea.scripts.generate_ai_variants --n-per-group 4 --strength 0.6
"""
import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[2]
COMBINED_CSV = BASE_DIR / "data/processed/bone_geometric_features_combined.csv"
OUT_DIR = BASE_DIR / "data/ai_generated_review"
MANIFEST_PATH = OUT_DIR / "manifest.csv"

MODEL_ID = "stabilityai/sd-turbo"
PROMPT = ("a real photograph of a bone specimen on a table, natural lighting, "
          "realistic surface texture, museum catalog photo, sharp focus")
NEGATIVE_PROMPT = "cartoon, drawing, illustration, 3d render, plastic, toy, blurry, text, watermark, cgi"
DEFAULT_STRENGTH = 0.5  # 0=igual al original, 1=ignora el original; sd-turbo funciona bien en 0.3-0.6
GUIDANCE_SCALE = 0.0  # sd-turbo se entrenó para generar sin CFG (guidance_scale=0)
NUM_INFERENCE_STEPS = 2  # sd-turbo: pensado para 1-4 pasos, no los ~30 de SD clásico


def sample_renders(n_per_group: int, seed: int) -> pd.DataFrame:
    """Un render por espécimen elegido al azar (no varias vistas del mismo
    espécimen) para maximizar diversidad de forma en la muestra, y solo de
    `renders/` (limpios) — partir de un render ya con ruido sería duplicar
    aumento sobre aumento sin necesidad."""
    df = pd.read_csv(COMBINED_CSV)
    df = df[df["source_path"].str.contains("/renders/", regex=False)]
    rng = random.Random(seed)
    rows = []
    for group, sub in df.groupby("bone_group"):
        specimens = sorted(sub["specimen"].unique().tolist())
        rng.shuffle(specimens)
        for specimen in specimens[:n_per_group]:
            candidates = sub[sub["specimen"] == specimen]
            rows.append(candidates.sample(1, random_state=seed).iloc[0])
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-group", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sample = sample_renders(args.n_per_group, args.seed)
    print(f"{len(sample)} renders elegidos ({args.n_per_group} especímenes/grupo x "
          f"{sample['bone_group'].nunique()} grupos)")

    manifest_rows = pd.read_csv(MANIFEST_PATH).to_dict("records") if MANIFEST_PATH.exists() else []
    already_done = {r["source_path"] for r in manifest_rows}
    sample = sample[~sample["source_path"].isin(already_done)]
    if sample.empty:
        print("Nada nuevo que generar (todos los elegidos ya están en el manifest).")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cargando {MODEL_ID} en {device}...")
    pipe = AutoPipelineForImage2Image.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    for _, row in sample.iterrows():
        source_path = row["source_path"]
        init_image = Image.open(source_path).convert("RGB").resize((512, 512))
        generated = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=init_image,
            strength=args.strength,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS,
        ).images[0]

        bone_group = row["bone_group"]
        out_dir = OUT_DIR / bone_group
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(source_path).stem}_ai.jpg"
        generated.save(out_path, quality=90)

        manifest_rows.append({
            "source_path": source_path,
            "generated_path": str(out_path),
            "bone_group": bone_group,
            "specimen": row["specimen"],
            "prompt": PROMPT,
            "strength": args.strength,
            "status": "pending",
        })
        print(f"  {Path(source_path).name} -> {out_path}")

    pd.DataFrame(manifest_rows).to_csv(MANIFEST_PATH, index=False)
    n_pending = sum(1 for r in manifest_rows if r["status"] == "pending")
    print(f"\nManifest: {MANIFEST_PATH} ({len(manifest_rows)} filas totales, {n_pending} pendientes de revisar)")
    print("Revisar con: streamlit run base_datos_osea/scripts/review_ai_generated.py --server.port 8502")


if __name__ == "__main__":
    main()
