# Persistencia: todo versionado, siempre

Regla fija del usuario (2026-10-27): cada cambio relevante de este proyecto tiene que quedar en
**git + GitHub + DVC + MLflow**, en el mismo momento en que se hace, no al final de la sesión.
"Relevante" = cualquier cosa que cueste reproducir: código, datasets procesados, modelos
entrenados, renders nuevos, resultados de examen. No vale dejarlo "ya lo subo luego" ni confiar en
que quede solo en disco local de esta máquina.

## Qué va en cada sistema

- **Git/GitHub**: código (`training/`, `scripts/`, `shared/`, `serving/`...), CSVs de
  `data/processed/` (se versionan directo en git, no por DVC, es el patrón ya establecido en este
  repo), `HISTORIAL.md`. Commit + push tras cada cambio con sentido propio, no al final del día.
- **DVC** (remoto `local-storage` -> `/root/dvc-storage`): todo lo demás que sea binario grande o
  no quepa en git razonablemente -- `models/*.pt`/`*.joblib`/`*.json` (hoy gitignored y SIN
  trackear en ningún sitio, bug de higiene detectado 2026-10-27), `base_datos_osea/renders_lab/`
  y cualquier carpeta de renders nueva, `data/segmented_crops*/`, `data/real_photos_*/`,
  `data/stage2_*_viewer/`. Tras generar/reentrenar: `dvc add <ruta>` -> `git add <ruta>.dvc` (y el
  `.gitignore` que DVC actualiza) -> commit -> `dvc push` + `git push`.
- **MLflow** (`http://127.0.0.1:5000`, backend `./mlruns`, necesita
  `MLFLOW_ALLOW_FILE_STORE=true` para arrancar en esta versión -- ver HISTORIAL 2026-10-27): cada
  entrenamiento (`train_cnn.py`) y cada examen real (`evaluate_cnn_real_photos.py`) ya loguea solo
  si el server está arriba -- comprobar que corre ANTES de lanzar cualquier training
  (`curl -s http://127.0.0.1:5000/version`), si no, levantarlo primero. Si no está levantado el
  entrenamiento revienta (no hay fallback silencioso), así que esto ya se cumple por diseño salvo
  que el server se caiga a media sesión.

## Cuándo hacerlo

Inmediatamente después de cada paso que produzca algo nuevo y reproducible, no acumulado para el
final:
- Tras construir/actualizar un dataset (`build_*.py`, `build_dataset_lab.py`,
  `build_segmented_crops.py`...): commit del CSV (git) o `dvc add` + push (si es carpeta de
  imágenes).
- Tras cada entrenamiento que se queda (no los descartados): `dvc add models/` de los ficheros
  nuevos/cambiados, commit+push, y confirmar que el run quedó en MLflow (ya pasa solo).
- Tras promover un modelo a default (copiar `..._seedN_...` sobre el nombre sin semilla): el commit
  de ese `dvc add models/` tiene que ir en el mismo lote que la entrada en `HISTORIAL.md` que
  documenta la promoción -- no como pasos separados en el tiempo.

## Qué NO hacer

- No dejar "para luego" un `dvc push`/`git push` tras tocar algo grande -- si la sesión se corta
  ahí, se pierde.
- No asumir que algo en `models/` o en una carpeta de renders está versionado porque "siempre se
  ha hecho así" -- comprobarlo (`dvc status`, `git ls-files <ruta>`) antes de asumirlo, el estado
  real a 2026-10-27 es que NO lo estaba.
