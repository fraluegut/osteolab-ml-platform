"""Particiones sin fuga entre vistas del mismo sujeto/especimen."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


SUBJECT_COLUMNS = ("species", "specimen")


def subject_ids(df: pd.DataFrame) -> np.ndarray:
    """Devuelve una clave estable de sujeto, evitando colisiones entre fuentes.

    ``specimen`` por si solo puede repetirse entre especies o colecciones. La
    combinación species+specimen mantiene juntas todas las vistas del mismo
    sujeto sin juntar accidentalmente sujetos distintos.
    """
    missing = [column for column in SUBJECT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para agrupar por sujeto: {missing}")
    if df[list(SUBJECT_COLUMNS)].isna().any().any():
        raise ValueError("Las columnas de sujeto no pueden contener valores nulos")
    return (
        df["species"].astype(str).str.strip()
        + "::"
        + df["specimen"].astype(str).str.strip()
    ).to_numpy()


def stratified_group_folds(
    df: pd.DataFrame,
    target_col: str,
    n_splits: int = 5,
    random_state: int = 42,
):
    """Genera folds estratificados, manteniendo cada sujeto en un solo lado."""
    groups = subject_ids(df)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Se necesitan al menos 2 sujetos para particionar")
    effective_splits = min(n_splits, len(unique_groups))
    if effective_splits < 2:
        raise ValueError("n_splits efectivo debe ser >= 2")
    splitter = StratifiedGroupKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=random_state,
    )
    yield from splitter.split(df, y=df[target_col].to_numpy(), groups=groups)


def subject_holdout(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Un holdout estratificado por sujeto para entrenamientos costosos."""
    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1")
    requested_splits = max(2, round(1 / test_size))
    train_idx, val_idx = next(
        stratified_group_folds(df, target_col, requested_splits, random_state)
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_subjects = set(subject_ids(train_df))
    val_subjects = set(subject_ids(val_df))
    overlap = train_subjects & val_subjects
    if overlap:
        raise RuntimeError(f"Fuga de sujeto entre train/val: {sorted(overlap)}")
    return train_df, val_df
