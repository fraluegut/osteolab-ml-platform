from pathlib import Path
from PIL import Image

DATA_DIR = Path("data/raw/bones")


def sanity_check():
    total_images = 0

    for class_dir in DATA_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*"))
        print(f"\nClase: {class_dir.name}")
        print(f"  Nº imágenes: {len(images)}")

        total_images += len(images)

        # Revisar primeras imágenes
        for img_path in images[:3]:
            try:
                img = Image.open(img_path)
                print(f"  ✔ {img_path.name} | tamaño: {img.size} | modo: {img.mode}")
            except Exception as e:
                print(f"  ❌ Error en {img_path.name}: {e}")

    print(f"\nTotal imágenes dataset: {total_images}")


if __name__ == "__main__":
    sanity_check()
    
"""
Clase: craneo
  Nº imágenes: 3630
  ✔ Craneo 2_BLENDER_EEVEE_Raw__True_242_img200.png | tamaño: (256, 256) | modo: RGB
  ✔ Craneo 4_BLENDER_EEVEE_Raw__True_242_img84.png | tamaño: (256, 256) | modo: RGB
  ✔ Craneo 1_BLENDER_EEVEE_Raw__True_242_img10.png | tamaño: (256, 256) | modo: RGB

Clase: humero
  Nº imágenes: 3630
  ✔ Humerus 4_CYCLES_Filmic__False_242_img189.png | tamaño: (256, 256) | modo: RGB
  ✔ Humerus 4_BLENDER_EEVEE_Standard__True_242_img200.png | tamaño: (256, 256) | modo: RGB
  ✔ Humerus 2_BLENDER_EEVEE_Raw__True_242_img200.png | tamaño: (256, 256) | modo: RGB

Clase: femur
  Nº imágenes: 3630
  ✔ Femur 5_CYCLES_Filmic__False_242_img25.png | tamaño: (256, 256) | modo: RGB
  ✔ Femur 3_CYCLES_Filmic__False_242_img106.png | tamaño: (256, 256) | modo: RGB
  ✔ Femur 5_BLENDER_EEVEE_Raw__True_242_img197.png | tamaño: (256, 256) | modo: RGB

Total imágenes dataset: 10890"""
