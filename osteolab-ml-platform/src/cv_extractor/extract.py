"""Localización del hueso en la imagen y extracción de medidas geométricas
con OpenCV (segmentación clásica, sin aprendizaje).

Módulo aislado: no depende de `src/clip_filter`, `src/bone_filter` ni
`src/inference`, y no comparte estado con ellos. Se invoca de forma explícita
desde el endpoint dedicado en `app/main.py`, después de que el filtro CLIP
confirme que la imagen contiene un hueso.

Las medidas que produce son relativas a la imagen (píxeles, ratios): no hay
forma de derivar medidas reales (cm) sin una referencia de escala en la
imagen, así que ese cálculo no se intenta aquí. El contexto que aporte el
usuario (medidas reales, peso, tipo...) se transporta por separado, sin
mezclarse con estas features.
"""
import base64
import io

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MIN_CONTOUR_AREA_RATIO = 0.001  # por debajo de esto, se considera "no encontrado"

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
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu no sabe si el objeto es la región clara o la oscura; asumimos que
    # el fondo ocupa más superficie que el objeto y, si el umbral deja más
    # píxeles "encendidos" que "apagados", el objeto es la región oscura.
    if cv2.countNonZero(binary) > binary.size / 2:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def _largest_contour(binary: np.ndarray):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


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


def extract_features(image_file) -> dict:
    """Devuelve:
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
          "annotated_image_base64": str,   # imagen con las medidas pegadas a cada lado
          "mask_overlay_base64": str,      # imagen con la región detectada resaltada
        }
    """
    data = np.frombuffer(image_file.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo decodificar la imagen")

    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = _segment(gray)
    contour = _largest_contour(binary)

    if contour is None or cv2.contourArea(contour) < img_area * MIN_CONTOUR_AREA_RATIO:
        return {"found": False, "image_size": {"width": img_w, "height": img_h}}

    contour_area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
    box_points = cv2.boxPoints(((cx, cy), (rw, rh), angle))

    long_side, short_side = max(rw, rh), min(rw, rh)
    hull_area = cv2.contourArea(cv2.convexHull(contour))

    shape_stats = {
        "area_ratio": round(float(contour_area / img_area), 6),
        "aspect_ratio": round(float(long_side / short_side), 4) if short_side > 0 else None,
        "extent": round(float(contour_area / (w * h)), 4) if w * h > 0 else None,
        "solidity": round(float(contour_area / hull_area), 4) if hull_area > 0 else None,
    }

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
        "annotated_image_base64": _dimensioned_image(
            image, contour, box_points, (x, y, w, h), (long_side, short_side, angle), shape_stats
        ),
        "mask_overlay_base64": _mask_overlay_image(image, binary),
    }
