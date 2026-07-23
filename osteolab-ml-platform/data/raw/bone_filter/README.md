# Dataset del filtro "¿es un hueso?"

Vuelca aquí las imágenes para entrenar el filtro binario aislado (ver
`src/bone_filter/`). No se mezcla con `data/processed/bone_geometric_features.csv`
(tabla del clasificador de grupo morfológico, ver `src/training/train_geometric.py`).

Estructura esperada (formato `ImageFolder` de torchvision):

```text
data/raw/bone_filter/
    bone/         # fotos de huesos: cualquier tipo (cráneo, fémur, húmero, radiografías, etc.)
    not_bone/     # fotos de cosas que NO son huesos (fondos, otros objetos, personas, animales enteros, etc.)
```

Recomendaciones:

- Formatos admitidos: `.jpg`, `.jpeg`, `.png`.
- Cuantas más imágenes y más variadas (ángulos, iluminación, fondos), mejor generaliza.
- Intenta un número similar de imágenes en `bone/` y `not_bone/` (dataset balanceado).
- Con unos pocos cientos de imágenes por clase ya se puede obtener un filtro razonable
  gracias al backbone preentrenado (ImageNet) de ResNet18.

Para entrenar una vez tengas las imágenes:

```bash
python src/bone_filter/train.py
```

El modelo entrenado se guarda en `models/bone_filter/` (no en `models/geometric_model_*.joblib`,
que pertenece al clasificador de grupo morfológico).
