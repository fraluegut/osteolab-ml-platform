"""Métricas extra compartidas por los scripts `evaluate_*.py` — más allá de
accuracy y el `classification_report` por clase que ya se calculaba, para
leer mejor el desempeño con clases desbalanceadas (pelvis: 100 filas vs
hueso_largo: 853) y grupos anatómicos parecidos entre sí (propuesta del
usuario, 2026-07-26).
"""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, top_k_accuracy_score


def extra_metrics(y_true_idx, y_pred_idx, y_proba: np.ndarray | None = None,
                   n_classes: int | None = None) -> dict:
    """`y_true_idx`/`y_pred_idx`: índices de clase (0..n-1). `y_proba`: matriz
    (n_samples, n_classes) opcional, en el mismo orden de clases que
    `n_classes` — sin ella se omite `top3_accuracy` (no todos los scripts
    tienen probabilidades a mano, p.ej. la cascada cuando decide por regla
    geométrica en vez de por CNN)."""
    metrics = {"balanced_accuracy": balanced_accuracy_score(y_true_idx, y_pred_idx)}
    if y_proba is not None and n_classes is not None:
        try:
            metrics["top3_accuracy"] = top_k_accuracy_score(
                y_true_idx, y_proba, k=min(3, n_classes), labels=list(range(n_classes)),
            )
        except ValueError:
            pass  # menos de 3 clases con ejemplos en este banco, no aplica
    return metrics


def format_confusion_matrix(y_true_idx, y_pred_idx, class_names) -> str:
    """Tabla de texto simple (filas=real, columnas=predicho) — pensada para
    imprimir en consola/loguear como artefacto de texto en MLflow, sin
    depender de matplotlib para algo que se va a leer en una terminal."""
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=range(len(class_names)))
    col_w = 8
    header = "real\\pred".ljust(20) + "".join(f"{c[:6]:>{col_w}}" for c in class_names)
    lines = [header]
    for i, row in enumerate(cm):
        lines.append(class_names[i][:19].ljust(20) + "".join(f"{v:>{col_w}}" for v in row))
    return "\n".join(lines)
