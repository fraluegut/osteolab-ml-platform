"""Localización del hueso en la imagen y extracción de medidas geométricas
con OpenCV (momentos de Hu, perfil de anchura, agujeros...) — la MEDICIÓN
sigue siendo clásica, sin aprendizaje.

La SEGMENTACIÓN (aislar el hueso del fondo, paso previo a medir) admite tres
backends: `"grounded_sam"` (Grounding DINO + SAM por texto, por defecto desde
2026-07-26 — el que sirve `serving/backend/main.py` hoy, ver `_segment_grounded_sam`),
`"fastsam"` (segmentación aprendida con un punto central, desde 2026-07-25,
ver `_segment_fastsam`) y `"classic"` (Otsu+adaptativo, el backend original)
— ver HISTORIAL.md para la comparación entre los tres sobre el banco de
fotos reales.

Módulo aislado: no depende de `shared/clip_filter`, `shared/bone_filter` ni
`shared/inference`, y no comparte estado con ellos. Se invoca de forma explícita
desde el endpoint dedicado en `serving/backend/main.py`, después de que el filtro CLIP
confirme que la imagen contiene un hueso.

Las medidas que produce son relativas a la imagen (píxeles, ratios): no hay
forma de derivar medidas reales (cm) sin una referencia de escala en la
imagen, así que ese cálculo no se intenta aquí. El contexto que aporte el
usuario (medidas reales, peso, tipo...) se transporta por separado, sin
mezclarse con estas features.
"""
import base64
import io
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MIN_CONTOUR_AREA_RATIO = 0.001  # por debajo de esto, se considera "no encontrado"

# Resolución de los renders de entrenamiento (ver base_datos_osea/blender/render_bone_noisy.py
# --resolution, siempre 512x512). Bug encontrado el 2026-07-25: ~37 de las 57 columnas de
# features son medidas en píxeles absolutos (área, bbox, perfil de anchura...), nunca ratios —
# sobre una foto real de varios miles de píxeles, esas columnas salen a decenas o miles de
# desviaciones típicas de lo visto en entrenamiento (llegó a medirse z=4096), sin que la FORMA
# real del hueso tenga nada que ver. Reescalar aquí, antes de medir nada, hace que "cuántos
# píxeles mide la foto" deje de colarse como señal — es un no-op sobre los renders de
# entrenamiento (ya son 512x512 exactos) así que no hace falta reextraer ni reentrenar nada.
CANONICAL_LONG_SIDE = 512

FONT_SIZE = 18
BOX_COLOR = (0, 170, 255, 255)  # bounding box vertical (ancho/alto)
ROTATED_COLOR = (255, 60, 60, 255)  # rectángulo rotado (orientación real)
CONTOUR_COLOR = (0, 220, 0, 255)
MASK_TINT = (255, 0, 180)


def _load_font() -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", FONT_SIZE)
    except OSError:
        return ImageFont.load_default(size=FONT_SIZE)


_FONT = _load_font()


def _segment(gray: np.ndarray) -> np.ndarray:
    """Combina un umbral global (Otsu) con uno local (adaptativo) antes de
    aislar el hueso.

    Motivación (2026-07-24, banco de fotos reales): Otsu solo funciona si el
    objeto tiene un tono uniforme distinto del fondo — cierto en los renders
    de entrenamiento (iluminación de estudio, fondo blanco liso) pero falso
    en una foto real, donde el propio hueso tiene sombreado/textura desigual
    y Otsu deja fuera media pieza (confirmado con fotos de usuario: solo
    ~45% del área real detectada). El umbral adaptativo sí capta esa
    variación local, pero por sí solo reacciona a cualquier micro-contraste
    (incluida la textura fina de la superficie del hueso) y sobre un render
    limpio, sin apenas contraste interno, solo marca el borde exterior, no
    el interior — por eso no sustituye a Otsu sin más, se combina con él.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu no sabe si el objeto es la región clara o la oscura; asumimos que
    # el fondo ocupa más superficie que el objeto y, si el umbral deja más
    # píxeles "encendidos" que "apagados", el objeto es la región oscura.
    if cv2.countNonZero(otsu) > otsu.size / 2:
        otsu = cv2.bitwise_not(otsu)

    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 5
    )
    combined = cv2.bitwise_or(otsu, adaptive)

    # Cierre agresivo: en fotos reales, lo que aporta el adaptativo suele ser
    # un patrón disperso (textura de la superficie), no un bloque sólido —
    # hace falta un kernel grande para fusionarlo en una silueta continua.
    big_kernel = np.ones((25, 25), np.uint8)
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, big_kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return closed
    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, [largest], -1, 255, thickness=-1)

    # Dentro de esa silueta rellena, restauramos como hueco (0) las zonas que
    # NI Otsu NI el adaptativo habían marcado — huecos anatómicos reales
    # (cuencas oculares, forámenes), no ruido de fondo, porque ya están
    # dentro del contorno externo detectado.
    holes = cv2.bitwise_and(filled, cv2.bitwise_not(combined))
    binary = cv2.bitwise_and(filled, cv2.bitwise_not(holes))

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SAM_MODEL_ID = "facebook/sam-vit-base"
GROUNDED_SAM_TEXT_PROMPT = "bone."
GROUNDED_SAM_BOX_THRESHOLD = 0.25
GROUNDED_SAM_TEXT_THRESHOLD = 0.2


@lru_cache(maxsize=1)
def _load_grounding_dino():
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL_ID).to(device)
    return processor, model


@lru_cache(maxsize=1)
def _load_sam():
    import torch
    from transformers import SamModel, SamProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
    model = SamModel.from_pretrained(SAM_MODEL_ID).to(device)
    return processor, model


def _segment_grounded_sam(image_bgr: np.ndarray) -> np.ndarray:
    """Segmentación por texto (2026-07-26): Grounding DINO detecta primero una
    caja delimitadora para el texto "bone" y SAM segmenta dentro de esa caja
    — a diferencia de `_segment_fastsam`, que asume que el hueso está
    centrado, esta no depende de la posición. Motivación: en huesos con
    hueco central (vértebra, sacro, pelvis), si el punto central de FastSAM
    cae dentro del foramen, segmenta el hueco en vez del hueso (ver
    HISTORIAL.md, caso `vertebra_05.jpg`) — encuadrar el objeto completo por
    semántica evita ese fallo. Más lento que FastSAM (dos modelos en vez de
    uno) pero, sobre GPU, sigue siendo ~0.5s/imagen.
    """
    import torch
    from PIL import Image as PILImage

    h, w = image_bgr.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pil_image = PILImage.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    dino_processor, dino_model = _load_grounding_dino()
    inputs = dino_processor(images=pil_image, text=GROUNDED_SAM_TEXT_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    detections = dino_processor.post_process_grounded_object_detection(
        outputs, threshold=GROUNDED_SAM_BOX_THRESHOLD, text_threshold=GROUNDED_SAM_TEXT_THRESHOLD,
        target_sizes=[(h, w)],
    )[0]
    if len(detections["boxes"]) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    best_idx = int(detections["scores"].argmax())
    box = detections["boxes"][best_idx].tolist()

    sam_processor, sam_model = _load_sam()
    sam_inputs = sam_processor(pil_image, input_boxes=[[box]], return_tensors="pt").to(device)
    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)
    masks = sam_processor.image_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(), sam_inputs["reshaped_input_sizes"].cpu(),
    )
    # SAM devuelve 3 máscaras candidatas (distinta granularidad) con un
    # iou_score que se autoasigna — NO usarlo para elegir: en fotos con fondo
    # casi sin contraste y una caja que cubre casi toda la imagen (caso real,
    # ver HISTORIAL.md 2026-07-26, `mandibula_01.jpg`), la máscara correcta
    # (silueta completa del hueso) puntuó más bajo (0.93) que dos sub-partes
    # erróneas (0.97 y 1.01) — SAM confía más en la lectura equivocada.
    # Elegir la de mayor ÁREA en su lugar: coincide con la mejor máscara en
    # todos los casos ya validados a ojo y evita ese fallo.
    candidate_masks = masks[0][0].numpy()
    best_mask_idx = int(candidate_masks.reshape(len(candidate_masks), -1).mean(axis=1).argmax())
    mask = (candidate_masks[best_mask_idx].astype(np.uint8)) * 255

    # Apertura morfológica: la máscara de SAM a veces trae un artefacto de
    # upsampling — una tira falsa de 1px pegada al borde de la imagen (caso
    # real, `hueso_plano_08.jpg`) que no cambia apenas el área en píxeles
    # pero sí "engancha" el contorno externo al borde, inflando
    # `cv2.contourArea` muy por encima del área real (0.52 → 0.98 medido).
    # La apertura la elimina sin tocar la silueta real.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _segment_grounded_sam_all(image_bgr: np.ndarray, text_prompt: str = GROUNDED_SAM_TEXT_PROMPT,
                               box_threshold: float = GROUNDED_SAM_BOX_THRESHOLD) -> list[np.ndarray]:
    """Como `_segment_grounded_sam` pero devuelve TODAS las cajas de Grounding
    DINO por encima del umbral (no solo `argmax`), cada una segmentada por SAM
    por separado — para fotos con varios huesos sueltos del mismo tipo en un
    plano (costillas en abanico/parrilla, ver HISTORIAL.md 2026-08-03: esas
    fotos se descartaban enteras por no poder recortarse limpio con un
    rectángulo, cuando en realidad cada hueso individual sí es segmentable).

    2026-08-03: usado por `segment_instances()` para producir un recorte por
    hueso en vez de descartar la imagen entera cuando hay varios.
    """
    import torch
    from PIL import Image as PILImage

    h, w = image_bgr.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pil_image = PILImage.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    dino_processor, dino_model = _load_grounding_dino()
    inputs = dino_processor(images=pil_image, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    detections = dino_processor.post_process_grounded_object_detection(
        outputs, threshold=box_threshold, text_threshold=GROUNDED_SAM_TEXT_THRESHOLD,
        target_sizes=[(h, w)],
    )[0]
    if len(detections["boxes"]) == 0:
        return []

    sam_processor, sam_model = _load_sam()
    masks_out = []
    for box in detections["boxes"].tolist():
        sam_inputs = sam_processor(pil_image, input_boxes=[[box]], return_tensors="pt").to(device)
        with torch.no_grad():
            sam_outputs = sam_model(**sam_inputs)
        masks = sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(), sam_inputs["reshaped_input_sizes"].cpu(),
        )
        candidate_masks = masks[0][0].numpy()
        best_mask_idx = int(candidate_masks.reshape(len(candidate_masks), -1).mean(axis=1).argmax())
        mask = (candidate_masks[best_mask_idx].astype(np.uint8)) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        masks_out.append(mask)
    return masks_out


def _dedupe_overlapping_masks(masks: list[np.ndarray], iou_threshold: float = 0.6) -> list[np.ndarray]:
    """Grounding DINO puede devolver varias cajas casi idénticas para el mismo
    hueso físico (una detección redundante con score algo distinto) — sin
    esto, `segment_instances` produciría 2 fotos casi iguales del mismo hueso
    en vez de una por hueso real. Se queda con la de mayor área entre cada par
    que se solapa por encima de `iou_threshold`."""
    keep = []
    areas = [int((m > 0).sum()) for m in masks]
    order = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)
    for i in order:
        mi = masks[i] > 0
        is_dup = False
        for j in keep:
            mj = masks[j] > 0
            inter = np.logical_and(mi, mj).sum()
            union = np.logical_or(mi, mj).sum()
            if union > 0 and inter / union > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)
    return [masks[i] for i in keep]


def segment_instances(image_file, text_prompt: str = GROUNDED_SAM_TEXT_PROMPT,
                       box_threshold: float = GROUNDED_SAM_BOX_THRESHOLD,
                       bg_gray: int = 128) -> list["Image.Image"]:
    """Para una foto con VARIOS huesos sueltos del mismo tipo (costillas en
    abanico/parrilla, vértebras en grid...): detecta cada uno por separado
    (Grounding DINO, todas las cajas por encima de umbral) y devuelve una
    lista de recortes — uno por hueso, con TODO lo demás (fondo + los otros
    huesos de la foto) pintado de gris neutro, igual que `segment_and_crop`
    pero aplicado hueso a hueso en vez de a la imagen entera.

    Uso: `python -m shared.cv_extractor.segment_instances_cli <carpeta>` (ver
    ese script) para procesar un lote y revisar visualmente antes de aceptar
    nada en el manifest — el recorte automático puede fallar si dos huesos
    están pegados/solapados de verdad, no solo próximos.
    """
    data = np.frombuffer(image_file.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo decodificar la imagen")
    image = _resize_to_canonical(image)

    masks = _segment_grounded_sam_all(image, text_prompt=text_prompt, box_threshold=box_threshold)
    img_area = image.shape[0] * image.shape[1]
    masks = [m for m in masks if (m > 0).sum() >= img_area * MIN_CONTOUR_AREA_RATIO]
    masks = _dedupe_overlapping_masks(masks)

    crops = []
    for mask in masks:
        contour = _largest_contour(mask)
        if contour is None:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        crops.append(_crop_masked(image, mask, x, y, w, h, bg_gray))
    return crops


FASTSAM_CHECKPOINT = "models/FastSAM-s.pt"


@lru_cache(maxsize=1)
def _load_fastsam():
    from ultralytics import FastSAM
    return FastSAM(FASTSAM_CHECKPOINT)


def _segment_fastsam(image_bgr: np.ndarray) -> np.ndarray:
    """Segmentación aprendida (2026-07-25): en vez de un umbral de intensidad
    (frágil ante sombras/textura de una foto real, ver `_segment`), usa
    FastSAM con un único prompt de punto en el centro de la imagen — asume
    que el hueso está más o menos centrado, cierto tanto en los renders de
    entrenamiento (encuadre de cámara fijo) como en la inmensa mayoría de
    fotos reales del banco. No requiere ninguna clase concreta ("hueso"): se
    limita a segmentar el objeto saliente bajo el punto, que es justo lo que
    hace falta después de que CLIP ya confirmó que hay un hueso en la imagen.
    Devuelve una máscara binaria (0/255) en el mismo formato que `_segment`.
    """
    import torch

    model = _load_fastsam()
    h, w = image_bgr.shape[:2]
    device = 0 if torch.cuda.is_available() else "cpu"
    results = model(image_bgr, points=[[w // 2, h // 2]], labels=[1], device=device, verbose=False)
    r = results[0]
    if r.masks is None or len(r.masks.data) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    mask = r.masks.data[0].detach().cpu().numpy()
    mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask_resized * 255).astype(np.uint8)


def _largest_contour(binary: np.ndarray):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _hole_features(binary: np.ndarray, contour_area: float) -> dict:
    """Agujeros internos del contorno externo (forámenes, forámenes
    nutricios...) — `_largest_contour` usa RETR_EXTERNAL a propósito para
    quedarse solo con el contorno de fuera, así que esta información se
    pierde por diseño en el resto del pipeline. Un hueso largo no tiene
    ninguno; una vértebra tiene uno central (el foramen vertebral); un sacro
    o una pelvis tienen varios — señal anatómica directa que Hu moments o
    aspect ratio no capturan, y que sobrevive en la mayoría (no todas: solo se
    ve un agujero pasante si el ángulo de cámara mira más o menos a través
    del conducto) de las vistas de un mismo hueso.

    Recalcula contornos con RETR_CCOMP (jerarquía de 2 niveles: externos +
    sus huecos directos) sobre el mismo binario ya segmentado — barato, y
    mantiene `_largest_contour` sin tocar para no afectar al resto de features.
    """
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or contour_area <= 0:
        return {"n_holes": 0, "holes_area_ratio": 0.0, "largest_hole_area_ratio": 0.0}

    areas = [cv2.contourArea(c) for c in contours]
    outer_idx = int(np.argmax(areas))
    # umbral relativo al tamaño del propio hueso en esta vista, no un nº de
    # píxeles fijo: descarta ruido de segmentación sin descartar forámenes
    # reales de huesos pequeños fotografiados de cerca.
    min_hole_area = contour_area * 0.0005
    hole_areas = [
        areas[i] for i, h in enumerate(hierarchy[0])
        if h[3] == outer_idx and areas[i] > min_hole_area
    ]
    total_hole_area = sum(hole_areas)

    return {
        "n_holes": len(hole_areas),
        "holes_area_ratio": round(total_hole_area / contour_area, 4),
        "largest_hole_area_ratio": round(max(hole_areas) / contour_area, 4) if hole_areas else 0.0,
    }


def _groove_features(contour: np.ndarray, size_ref: float) -> dict:
    """Defectos de convexidad (cuánto se hunde el contorno respecto a su
    envolvente convexa, en cada muesca) como proxy 2D de escotaduras,
    trócleas y cóndilos articulares. `solidity`/`convexity` ya usan el hull
    pero resumen toda la envolvente en un único número — no distinguen "una
    muesca profunda" de "muchas muescas superficiales", que es justo la
    diferencia entre, por ejemplo, un hueso con una superficie articular
    marcada y uno sin ella.

    `size_ref` normaliza la profundidad (viene en píxeles) por una medida
    propia del hueso en esa vista (el lado largo del rectángulo rotado), para
    que no dependa de la resolución de la imagen ni del tamaño real del hueso.
    """
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 4 or size_ref <= 0:
        return {"n_deep_grooves": 0, "max_groove_depth": 0.0, "mean_groove_depth": 0.0}

    hull_indices = np.sort(hull_indices, axis=0)
    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        defects = None
    if defects is None:
        return {"n_deep_grooves": 0, "max_groove_depth": 0.0, "mean_groove_depth": 0.0}

    # cv2.convexityDefects devuelve (N, 4) en OpenCV >=5 y (N, 1, 4) en
    # versiones anteriores — reshape defensivo para que funcione con ambas.
    depths = (defects.reshape(-1, 4)[:, 3].astype(np.float64) / 256.0) / size_ref
    deep = depths[depths > 0.02]  # hundimiento >2% del lado largo del hueso

    return {
        "n_deep_grooves": int(len(deep)),
        "max_groove_depth": round(float(depths.max()), 4),
        "mean_groove_depth": round(float(deep.mean()), 4) if len(deep) else 0.0,
    }


def _elongation(contour: np.ndarray) -> float:
    """1 - (autovalor menor / autovalor mayor) de la nube de puntos del
    contorno: 0 para una forma compacta (círculo), cerca de 1 cuanto más
    alargada, sin depender de que esté alineada a los ejes."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    centered = pts - pts.mean(axis=0)
    cov = (centered.T @ centered) / len(pts)
    eigvals = np.linalg.eigvalsh(cov)
    lam_min, lam_max = float(eigvals[0]), float(eigvals[-1])
    return round(1 - (lam_min / lam_max), 4) if lam_max > 1e-9 else 0.0


def _ellipse_ratio(contour: np.ndarray) -> float | None:
    """Eje mayor / eje menor de la elipse ajustada al contorno (necesita al
    menos 5 puntos)."""
    if len(contour) < 5:
        return None
    (_, _), (ma, mb), _ = cv2.fitEllipse(contour)
    short_axis, long_axis = sorted((ma, mb))
    return round(long_axis / short_axis, 4) if short_axis > 1e-6 else None


def _hu_moments(contour: np.ndarray) -> dict:
    """Los 7 momentos de Hu (invariantes a traslación, escala y rotación),
    con la transformación logarítmica estándar (-sign(h)*log10(|h|)) para que
    sean comparables entre sí: los valores en crudo abarcan órdenes de
    magnitud muy distintos y son inutilizables tal cual."""
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    log_hu = [(-np.sign(h) * np.log10(abs(h))) if abs(h) > 1e-12 else 0.0 for h in hu]
    return {f"hu{i + 1}": round(float(v), 4) for i, v in enumerate(log_hu)}


def _menger_curvature(p1, p2, p3) -> float:
    """Curvatura (1/radio del círculo que pasa por los 3 puntos) en el punto
    central; 0 si los puntos están (casi) alineados."""
    p1, p2, p3 = np.asarray(p1), np.asarray(p2), np.asarray(p3)
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])) / 2
    denom = a * b * c
    return 4 * area / denom if denom > 1e-9 else 0.0


def _signed_turning_angle(p1, p2, p3) -> float:
    """Ángulo de giro (con signo) entre el segmento p1->p2 y p2->p3, en
    radianes — positivo si gira a la izquierda, negativo a la derecha. A
    diferencia de `_menger_curvature` (magnitud, sin signo), esto permite
    distinguir "gira siempre hacia el mismo lado" (arco, como una costilla) de
    "gira para un lado y para otro" (recta con ruido, o forma en S)."""
    v1 = np.asarray(p2) - np.asarray(p1)
    v2 = np.asarray(p3) - np.asarray(p2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    angle = float(np.arccos(cos_a))
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return angle if cross >= 0 else -angle


def _fit_circle_residual(points: np.ndarray) -> float | None:
    """Ajuste algebraico de círculo (Kåsa) a `points` — devuelve el residuo
    RMS normalizado por el radio ajustado (0 = encaja perfecto en un arco de
    circunferencia, mayor cuanto más se aleje esa forma de un arco limpio).
    `None` si el ajuste degenera (puntos casi colineales, radio no finito)."""
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0] / 2, sol[1] / 2
    r_sq = sol[2] + cx ** 2 + cy ** 2
    if r_sq <= 1e-6:
        return None
    r = float(np.sqrt(r_sq))
    dists = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    residual = float(np.sqrt(np.mean((dists - r) ** 2)))
    return residual / r if r > 1e-6 else None


def _fit_line_residual(points: np.ndarray) -> float:
    """Distancia RMS de `points` a la recta que mejor ajusta (mínimos
    cuadrados) — 0 si son colineales, mayor cuanto más se curve la forma.
    Los puntos ya vienen en coordenadas (a lo largo del eje principal,
    desviación perpendicular), así que un ajuste simple y = a·x + b es
    suficiente sin necesitar rotación adicional."""
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ sol
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def _axis_shape_features(centerline: list[tuple[float, float]]) -> dict:
    """Propuesta del usuario (2026-08-03, pensada para separar costilla —muy
    tortuosa, gira siempre igual— de huesos largos —casi rectos—):
    tortuosidad, giro acumulado, consistencia del giro, y ajuste a
    arco/línea. Todo calculado sobre la misma `centerline` que ya produce
    `_width_profile_and_curvature` (sin segmentar ni muestrear de nuevo)."""
    if len(centerline) < 3:
        return {
            "tortuosity": 0.0, "total_turning_angle": 0.0, "turning_consistency": 0.0,
            "circle_fit_residual": None, "line_fit_residual": 0.0,
        }

    pts = np.asarray(centerline, dtype=np.float64)
    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    arc_length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    tortuosity = round(arc_length / chord, 4) if chord > 1e-6 else 0.0

    signed_angles = [
        _signed_turning_angle(centerline[i - 1], centerline[i], centerline[i + 1])
        for i in range(1, len(centerline) - 1)
    ]
    total_abs_turning = sum(abs(a) for a in signed_angles)
    # consistencia: 1.0 = gira siempre hacia el mismo lado (arco limpio, como
    # una costilla); 0.0 = gira igual a un lado que al otro (recta con ruido,
    # o una forma en S) — el signo se cancela al sumar en ese caso.
    turning_consistency = (
        round(abs(sum(signed_angles)) / total_abs_turning, 4) if total_abs_turning > 1e-9 else 0.0
    )

    circle_residual = _fit_circle_residual(pts)
    line_residual = _fit_line_residual(pts)

    return {
        "tortuosity": tortuosity,
        "total_turning_angle": round(total_abs_turning, 4),
        "turning_consistency": turning_consistency,
        "circle_fit_residual": round(circle_residual, 4) if circle_residual is not None else None,
        "line_fit_residual": round(line_residual, 4),
    }


def _width_profile_and_curvature(contour: np.ndarray, box_points: np.ndarray, n_samples: int = 20) -> dict:
    """Perfil de anchura a lo largo del eje principal del hueso (n_samples
    cortes perpendiculares equiespaciados) y curvatura de la línea central
    que resulta de unir el punto medio de cada corte. Usa el eje principal
    del rectángulo rotado, no los ejes de la imagen, así que funciona igual
    esté el hueso derecho o inclinado."""
    box = np.asarray(box_points, dtype=np.float64)
    edge_lengths = [np.linalg.norm(box[i] - box[i - 1]) for i in range(4)]
    long_edge = int(np.argmax(edge_lengths))
    u = (box[long_edge] - box[long_edge - 1])
    u = u / np.linalg.norm(u)
    v = np.array([-u[1], u[0]])

    pts = contour.reshape(-1, 2).astype(np.float64)
    center = box.mean(axis=0)
    local = pts - center
    x = local @ u
    y = local @ v

    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-6:
        empty_profile = [0.0] * n_samples
        return {
            "width_profile": empty_profile, "mean_width": 0.0, "std_width": 0.0,
            "max_width": 0.0, "min_width": 0.0, "curvature_mean": 0.0, "curvature_max": 0.0,
            "tortuosity": 0.0, "total_turning_angle": 0.0, "turning_consistency": 0.0,
            "circle_fit_residual": None, "line_fit_residual": 0.0,
        }

    sample_xs = np.linspace(x_min, x_max, n_samples)
    band = (x_max - x_min) / (n_samples * 1.5)

    widths, centerline = [], []
    for sx in sample_xs:
        mask = np.abs(x - sx) <= band
        if not np.any(mask):
            mask = np.argsort(np.abs(x - sx))[:5]
        y_slice = y[mask]
        widths.append(float(y_slice.max() - y_slice.min()))
        centerline.append((float(sx), float((y_slice.max() + y_slice.min()) / 2)))

    widths_arr = np.array(widths)
    curvatures = [
        _menger_curvature(centerline[i - 1], centerline[i], centerline[i + 1])
        for i in range(1, len(centerline) - 1)
    ]

    # `centerline` viene del punto medio de cada corte de anchura — sensible al
    # ruido del borde de la máscara (foto real, artefactos de SAM), así que un
    # tramo perfectamente recto puede mostrar zigzag punto a punto que no es
    # curvatura real del hueso, solo ruido de medición. `curvature_mean/max` ya
    # existía y se deja tal cual (no tocar su comportamiento ya validado); para
    # las métricas nuevas de tortuosidad/giro se usa una versión suavizada
    # (media móvil en la desviación perpendicular, la coordenada con ruido —
    # la posición a lo largo del eje ya es uniforme por construcción).
    ys = np.array([c[1] for c in centerline])
    if len(ys) >= 5:
        kernel = np.ones(3) / 3
        ys_smooth = np.convolve(ys, kernel, mode="same")
        ys_smooth[0], ys_smooth[-1] = ys[0], ys[-1]
    else:
        ys_smooth = ys
    smoothed_centerline = [(c[0], float(ys_smooth[i])) for i, c in enumerate(centerline)]

    return {
        "width_profile": [round(w, 2) for w in widths],
        "mean_width": round(float(widths_arr.mean()), 2),
        "std_width": round(float(widths_arr.std()), 2),
        "max_width": round(float(widths_arr.max()), 2),
        "min_width": round(float(widths_arr.min()), 2),
        "curvature_mean": round(float(np.mean(curvatures)), 6) if curvatures else 0.0,
        "curvature_max": round(float(np.max(curvatures)), 6) if curvatures else 0.0,
        **_axis_shape_features(smoothed_centerline),
    }


def _text_with_outline(draw: ImageDraw.ImageDraw, xy, text, anchor="la", fill=(255, 255, 0, 255)):
    """Texto con borde negro para que se lea sobre cualquier fondo."""
    x, y = xy
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                draw.text((x + dx, y + dy), text, font=_FONT, fill=(0, 0, 0, 255), anchor=anchor)
    draw.text((x, y), text, font=_FONT, fill=fill, anchor=anchor)


def _encode_png(image_rgba: Image.Image) -> str:
    buf = io.BytesIO()
    image_rgba.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _dimensioned_image(image_bgr, contour, box_points, bbox, rotated_dims, shape_stats) -> str:
    """Imagen anotada con las medidas pegadas junto a cada lado (ancho junto
    al ancho, alto junto al alto) y un panel con el resto de estadísticas."""
    x, y, w, h = bbox
    long_side, short_side, angle = rotated_dims
    img_w, img_h = image_bgr.shape[1], image_bgr.shape[0]

    base = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(base)

    # Contorno detectado
    contour_pts = [tuple(int(v) for v in p[0]) for p in contour]
    draw.line(contour_pts + [contour_pts[0]], fill=CONTOUR_COLOR, width=3)

    # Bounding box vertical, con el ancho y el alto escritos junto a cada lado
    # (anchor="mb"/"lm" deja el texto pegado pero fuera de la línea, sin pisarla)
    draw.rectangle([x, y, x + w, y + h], outline=BOX_COLOR, width=2)
    box_color_opaque = BOX_COLOR[:3] + (255,)

    ancho_text = f"Ancho: {w}px"
    ancho_half_w = _FONT.getlength(ancho_text) / 2 + 4
    ancho_cx = min(max(x + w / 2, ancho_half_w), img_w - ancho_half_w)
    _text_with_outline(draw, (ancho_cx, max(y - 6, FONT_SIZE)), ancho_text, anchor="mb", fill=box_color_opaque)

    alto_text = f"Alto: {h}px"
    alto_text_w = _FONT.getlength(alto_text)
    space_right = img_w - (x + w)
    if space_right >= alto_text_w + 15:
        alto_xy, alto_anchor = (x + w + 10, y + h / 2), "lm"
    else:
        alto_xy, alto_anchor = (max(x - 10, alto_text_w + 5), y + h / 2), "rm"
    _text_with_outline(draw, alto_xy, alto_text, anchor=alto_anchor, fill=box_color_opaque)

    # Rectángulo rotado: la orientación real del hueso, no necesariamente alineada a los ejes
    rbox_pts = [tuple(float(v) for v in p) for p in box_points]
    draw.line(rbox_pts + [rbox_pts[0]], fill=ROTATED_COLOR, width=2)
    rotado_text = f"Rotado: {long_side:.0f}×{short_side:.0f}px ({angle:.0f}°)"
    rotado_half_w = _FONT.getlength(rotado_text) / 2 + 4
    rcx = sum(p[0] for p in rbox_pts) / 4
    rcx = min(max(rcx, rotado_half_w), img_w - rotado_half_w)
    rbox_bottom = max(p[1] for p in rbox_pts)
    _text_with_outline(
        draw, (rcx, min(rbox_bottom + 8, img_h - FONT_SIZE - 4)),
        rotado_text,
        anchor="ma", fill=ROTATED_COLOR[:3] + (255,),
    )

    # Franja de estadísticas debajo de la foto, en un lienzo ampliado y con
    # una línea por estadística: así nunca se solapa ni se corta, sea cual
    # sea el tamaño de la imagen.
    lines = [
        f"Área ocupada: {shape_stats['area_ratio'] * 100:.1f}%",
        f"Ratio de aspecto: {shape_stats['aspect_ratio']}",
        f"Extensión: {shape_stats['extent']}",
        f"Solidez: {shape_stats['solidity']}",
    ]
    line_h = FONT_SIZE + 6
    strip_h = len(lines) * line_h + 10
    canvas = Image.new("RGBA", (img_w, img_h + strip_h), (20, 20, 20, 255))
    canvas.paste(base, (0, 0))
    strip_draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        strip_draw.text((10, img_h + 5 + i * line_h), line, font=_FONT, fill=(255, 255, 255, 255))

    return _encode_png(canvas)


def _mask_overlay_image(image_bgr, binary_mask) -> str:
    """Imagen original con la región que OpenCV considera "hueso" resaltada,
    para poder verificar visualmente si la segmentación acertó."""
    base = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    tint = Image.new("RGBA", base.size, MASK_TINT + (0,))
    alpha = Image.fromarray(binary_mask).convert("L").point(lambda v: 110 if v > 0 else 0)
    tint.putalpha(alpha)
    composed = Image.alpha_composite(base, tint)
    return _encode_png(composed)


def _resize_to_canonical(image: np.ndarray, target_long_side: int = CANONICAL_LONG_SIDE) -> np.ndarray:
    """Reescala conservando proporción (sin recortar ni deformar) para que el lado mayor mida
    `target_long_side` px, en ambas direcciones (fotos más grandes O más pequeñas que el
    objetivo) — ver el porqué en `CANONICAL_LONG_SIDE`. Deformar el aspect ratio distorsionaría
    los momentos de Hu/elongación/etc (se calculan sobre las coordenadas del contorno), por eso
    aquí se escala igual en X e Y en vez de forzar un lienzo cuadrado."""
    h, w = image.shape[:2]
    scale = target_long_side / max(h, w)
    if scale == 1.0:
        return image
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def _locate_bone(image: np.ndarray, backend: str):
    """Segmenta `image` (ya en tamaño canónico) y devuelve `(binary, contour)`
    — o `(None, None)` si no se encuentra nada por encima de
    `MIN_CONTOUR_AREA_RATIO`. Compartido por `extract_features` y
    `segment_and_crop` para no correr la segmentación dos veces sobre la
    misma imagen en el mismo sitio."""
    if backend == "none":
        # 2026-10-31: baseline "sin segmentar" para medir honestamente cuánto
        # aporta segmentar -- máscara = imagen entera, sin recorte ni fondo
        # pintado de gris. Comparación justa: el modelo que se evalúa así
        # también se ENTRENA así (ver train_cnn.py --dataset-path con el CSV
        # "sin segmentar"), no es el modelo de producción (entrenado con
        # recortes) alimentado con algo que nunca vio.
        binary = np.full(image.shape[:2], 255, dtype=np.uint8)
    elif backend == "fastsam":
        binary = _segment_fastsam(image)
    elif backend == "grounded_sam":
        binary = _segment_grounded_sam(image)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = _segment(gray)
    contour = _largest_contour(binary)
    img_area = image.shape[0] * image.shape[1]
    if contour is None or cv2.contourArea(contour) < img_area * MIN_CONTOUR_AREA_RATIO:
        return None, None
    return binary, contour


CROP_BG_GRAY = 128  # gris neutro fuera de la máscara — mismo valor en los 3 canales


def _crop_masked(image: np.ndarray, binary: np.ndarray, x: int, y: int, w: int, h: int,
                  bg_gray: int = CROP_BG_GRAY) -> Image.Image:
    """Recorte al bounding box `(x, y, w, h)` con los píxeles fuera de
    `binary` pintados de gris neutro — pensada para alimentar la CNN sin que
    el fondo (real o sintético) llegue a influir en absoluto (decisión del
    2026-07-26, ver HISTORIAL.md: hasta ahora la CNN veía siempre la imagen
    completa sin segmentar, solo el vector de features geométricas pasaba
    por la máscara)."""
    mask = binary > 0
    bg = np.full_like(image, bg_gray)
    masked = np.where(mask[..., None], image, bg)
    crop = masked[y:y + h, x:x + w]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def segment_and_crop(image_file, backend: str = "grounded_sam", bg_gray: int = CROP_BG_GRAY) -> Image.Image | None:
    """Segmenta una imagen suelta y devuelve el recorte de `_crop_masked`, o
    `None` si no se encuentra contorno — el llamador decide qué hacer
    (descartar la fila, normalmente). Para cuando ya se tiene `extract_features(..., include_crop=True)`
    corrido, usar el `"crop_image"` de su resultado en vez de llamar a esto
    (evita segmentar dos veces la misma imagen)."""
    data = np.frombuffer(image_file.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo decodificar la imagen")
    image = _resize_to_canonical(image)

    binary, contour = _locate_bone(image, backend)
    if binary is None:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    return _crop_masked(image, binary, x, y, w, h, bg_gray)


def extract_features(image_file, backend: str = "grounded_sam", include_crop: bool = False,
                      bg_gray: int = CROP_BG_GRAY) -> dict:
    """`backend`: "grounded_sam" (segmentación aprendida por texto, por
    defecto desde 2026-07-26 — ver `_segment_grounded_sam` y HISTORIAL.md),
    "fastsam" (segmentación aprendida por punto central, ver
    `_segment_fastsam`) o "classic" (Otsu+adaptativo, el backend original).

    Devuelve:
        {
          "found": bool,
          "image_size": {"width": int, "height": int},
          "bounding_box_px": {"x", "y", "width", "height"},      # solo si found
          "rotated_box_px": {"width", "height", "angle_deg"},     # solo si found
          "area_px": float,
          "area_ratio": float,      # área del contorno / área de la imagen
          "aspect_ratio": float,    # lado largo / lado corto del rectángulo rotado
          "extent": float,          # área del contorno / área del bounding box
          "solidity": float,        # área del contorno / área de su envolvente convexa
          "elongation": float,      # 1 - autovalores de la nube de puntos (PCA), 0=compacto..1=alargado
          "ellipse_ratio": float,   # eje mayor / eje menor de la elipse ajustada
          "circularity": float,     # 4π·área / perímetro² (1.0 = círculo perfecto)
          "convexity": float,       # perímetro de la envolvente convexa / perímetro del contorno
          "hu_moments": {"hu1".."hu7": float},   # invariantes de forma, escala log
          "mean_width": float, "std_width": float, "max_width": float, "min_width": float,
          "curvature_mean": float, "curvature_max": float,   # de la línea central (1/px)
          "width_profile": [float] * 20,   # anchura en 20 cortes a lo largo del eje principal
          "tortuosity": float,       # longitud del eje / distancia entre extremos (1.0 = recto)
          "total_turning_angle": float,   # suma de los giros del eje, en radianes (0 = recto)
          "turning_consistency": float,   # 0-1, 1.0 = gira siempre hacia el mismo lado (arco limpio)
          "circle_fit_residual": float | None,   # RMS al mejor círculo ajustado, /radio (None si degenera)
          "line_fit_residual": float,     # RMS a la mejor recta ajustada al eje
          "n_holes": int, "holes_area_ratio": float, "largest_hole_area_ratio": float,  # forámenes/agujeros internos
          "n_deep_grooves": int, "max_groove_depth": float, "mean_groove_depth": float,  # muescas articulares (defectos de convexidad)
          "annotated_image_base64": str,   # imagen con las medidas pegadas a cada lado
          "mask_overlay_base64": str,      # imagen con la región detectada resaltada
        }
    """
    data = np.frombuffer(image_file.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo decodificar la imagen")
    image = _resize_to_canonical(image)

    binary, contour = _locate_bone(image, backend)
    return _compute_all_features(image, binary, contour, include_crop, bg_gray)


def extract_features_from_mask(image_file, mask_file, include_crop: bool = False,
                                bg_gray: int = CROP_BG_GRAY) -> dict:
    """Igual que `extract_features`, pero con la máscara de segmentación YA
    dada (p.ej. la máscara ground-truth que exporta BlenderProc desde la
    propia geometría 3D, ver `render_bone_bproc.py` y HISTORIAL.md
    2026-07-26) en vez de adivinarla con un backend de segmentación sobre
    los píxeles. Evita el problema medido de huesos pequeños/lejanos y bajo
    contraste que Otsu/FastSAM/grounded_sam fallan en detectar (~12% de
    renders limpios descartados por esto) -- con la máscara exacta, esos
    casos ya no se pierden."""
    data = np.frombuffer(image_file.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo decodificar la imagen")
    image = _resize_to_canonical(image)

    mask_data = np.frombuffer(mask_file.read(), dtype=np.uint8)
    mask = cv2.imdecode(mask_data, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("No se pudo decodificar la máscara")
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary = (mask > 127).astype(np.uint8) * 255
    contour = _largest_contour(binary)
    img_area = image.shape[0] * image.shape[1]
    if contour is None or cv2.contourArea(contour) < img_area * MIN_CONTOUR_AREA_RATIO:
        binary, contour = None, None
    return _compute_all_features(image, binary, contour, include_crop, bg_gray)


def _compute_all_features(image: np.ndarray, binary, contour, include_crop: bool,
                           bg_gray: int = CROP_BG_GRAY) -> dict:
    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w

    if binary is None:
        return {"found": False, "image_size": {"width": img_w, "height": img_h}}

    contour_area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    box_points = cv2.boxPoints(((cx, cy), (rw, rh), angle))

    long_side, short_side = max(rw, rh), min(rw, rh)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)

    shape_stats = {
        "area_ratio": round(float(contour_area / img_area), 6),
        "aspect_ratio": round(float(long_side / short_side), 4) if short_side > 0 else None,
        "extent": round(float(contour_area / (w * h)), 4) if w * h > 0 else None,
        "solidity": round(float(contour_area / hull_area), 4) if hull_area > 0 else None,
        "elongation": _elongation(contour),
        "ellipse_ratio": _ellipse_ratio(contour),
        "circularity": round(float(4 * np.pi * contour_area / perimeter**2), 4) if perimeter > 0 else None,
        "convexity": round(float(hull_perimeter / perimeter), 4) if perimeter > 0 else None,
    }
    width_stats = _width_profile_and_curvature(contour, box_points)
    hole_stats = _hole_features(binary, contour_area)
    groove_stats = _groove_features(contour, long_side)

    return {
        "found": True,
        "image_size": {"width": img_w, "height": img_h},
        "bounding_box_px": {"x": x, "y": y, "width": w, "height": h},
        "rotated_box_px": {
            "width": round(float(long_side), 2),
            "height": round(float(short_side), 2),
            "angle_deg": round(float(angle), 2),
        },
        "area_px": round(float(contour_area), 2),
        **shape_stats,
        "hu_moments": _hu_moments(contour),
        **width_stats,
        **hole_stats,
        **groove_stats,
        "annotated_image_base64": _dimensioned_image(
            image, contour, box_points, (x, y, w, h), (long_side, short_side, angle), shape_stats
        ),
        "mask_overlay_base64": _mask_overlay_image(image, binary),
        **({"crop_image": _crop_masked(image, binary, x, y, w, h, bg_gray)} if include_crop else {}),
    }
