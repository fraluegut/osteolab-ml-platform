import pandas as pd

from training.group_splits import stratified_group_folds, subject_holdout, subject_ids


def _dataset():
    rows = []
    for label in ("a", "b"):
        for specimen in range(5):
            for view in range(3):
                rows.append({
                    "species": "human",
                    "specimen": f"{label}-{specimen}",
                    "bone_group": label,
                    "view": view,
                })
    return pd.DataFrame(rows)


def test_subject_id_uses_species_to_avoid_collisions():
    df = pd.DataFrame({"species": ["human", "chimp"], "specimen": ["1", "1"]})
    assert len(set(subject_ids(df))) == 2


def test_cross_validation_never_leaks_subjects():
    df = _dataset()
    for train_idx, val_idx in stratified_group_folds(df, "bone_group", n_splits=5):
        train_subjects = set(subject_ids(df.iloc[train_idx]))
        val_subjects = set(subject_ids(df.iloc[val_idx]))
        assert train_subjects.isdisjoint(val_subjects)
        assert set(df.iloc[val_idx]["bone_group"]) == {"a", "b"}


def test_holdout_never_leaks_subjects():
    df = _dataset()
    train_df, val_df = subject_holdout(df, "bone_group", test_size=0.2)
    assert set(subject_ids(train_df)).isdisjoint(subject_ids(val_df))
