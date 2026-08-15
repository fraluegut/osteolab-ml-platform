"""Cascada de 2 etapas sobre `predict_cnn.classify_image`: etapa 1 predice
`bone_group` (el `hybrid` de siempre, sin tocar) y, si el grupo predicho
tiene un clasificador de etapa 2 (hueso concreto DENTRO de ese grupo), se
llama a ese modelo aparte para afinar. Confirmado en
`base_datos_osea/HISTORIAL.md` (2026-08-14): un único clasificador plano con
más clases interfiere entre grupos no relacionados (`hueso_plano` colapsaba
a 0.00 recall al sumarle fémur/húmero/astrágalo); separar cada hueso
concreto en su propio modelo, entrenado solo con las filas de su grupo,
evita esa interferencia y además separa mejor el hueso concreto en sí.

Cada etapa 2 se entrena con `train_cnn.py --target bone_fine` sobre un CSV
filtrado a un solo `bone_group` (ver HISTORIAL.md) y se guarda con
`--run-name stage2_<grupo>`.

`taxon` (humano/no_humano, 2026-08-14) es una salida PARALELA a `fine_bone`,
no secuencial con ella -- ningún grupo tiene hoy ambas cosas a la vez
(cráneo/mandíbula_maxilar tienen taxón pero no hueso fino; hueso_largo/
hueso_pequeno/vertebra tienen hueso fino pero no taxón todavía). Si algún
grupo llega a necesitar las dos, lo natural es fusionarlas en un solo
modelo multi-cabeza entrenado junto (ver HISTORIAL.md 2026-08-14) en vez de
encadenarlas -- evita el problema de error que se arrastra en cascada Y dejar
huecos de datos aún más finos (hueso × taxón) que los que ya costó llenar.
"""
from shared.inference.predict_cnn import classify_image

# grupo de etapa 1 -> run_name del modelo de etapa 2 que afina el hueso concreto
STAGE2_MODELS = {
    "hueso_largo": "stage2_hueso_largo",
    "hueso_pequeno": "stage2_hueso_pequeno",
    "vertebra": "stage2_vertebra",
}

# grupo de etapa 1 -> run_name del modelo de etapa 2 que afina el taxón (humano/no_humano)
STAGE2_TAXON_MODELS = {
    "cranio": "stage2_taxon_cranio",
    "mandibula_maxilar": "stage2_taxon_mandibula_maxilar",
}


def classify_image_two_stage(image_file, backend: str = "grounded_sam", bg_gray: int = 128) -> dict:
    """Igual que `predict_cnn.classify_image(model_type="hybrid")`, con dos
    claves más: `"fine_bone"` y `"taxon"` (`None` si el grupo predicho no
    tiene el modelo de etapa 2 correspondiente, o no se pudo determinar).

    Devuelve el mismo dict que `classify_image` (etapa 1) más:
        "fine_bone": {"prediction": str, "probabilities": {...}} | None
        "taxon": {"prediction": str, "probabilities": {...}} | None
    """
    result = classify_image(image_file, model_type="hybrid", backend=backend, bg_gray=bg_gray)
    result["fine_bone"] = None
    result["taxon"] = None
    if not result["found"]:
        return result

    group = result["prediction"]

    stage2_run_name = STAGE2_MODELS.get(group)
    if stage2_run_name is not None:
        image_file.seek(0)
        fine = classify_image(image_file, model_type="hybrid", backend=backend, bg_gray=bg_gray,
                               run_name=stage2_run_name, target="bone_fine")
        if fine["found"]:
            result["fine_bone"] = {
                "prediction": fine["prediction"],
                "probabilities": fine["probabilities"],
            }

    taxon_run_name = STAGE2_TAXON_MODELS.get(group)
    if taxon_run_name is not None:
        image_file.seek(0)
        taxon = classify_image(image_file, model_type="hybrid", backend=backend, bg_gray=bg_gray,
                                run_name=taxon_run_name, target="taxon")
        if taxon["found"]:
            result["taxon"] = {
                "prediction": taxon["prediction"],
                "probabilities": taxon["probabilities"],
            }

    return result
