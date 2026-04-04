import sys, os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except:
    BASE_DIR = Path.cwd()

sys.path.insert(0, str(BASE_DIR))

from configs.config import (
    DATA_PROC, SPLITS, CENTERS,
    TEST_RATIO_SEG, TEST_RATIO_CLS,
    VAL_RATIO, RANDOM_SEED, CV_FOLDS
)

os.makedirs(SPLITS, exist_ok=True)


def shuffle_df(df):
    return df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def patient_level_df(df):
    return (
        df.groupby("patient")["label"]
        .agg(lambda x: int(x.mode()[0]))
        .reset_index()
    )


def safe_concat(dfs):
    dfs = [d for d in dfs if len(d) > 0]
    if len(dfs) == 0:
        return pd.DataFrame()
    return shuffle_df(pd.concat(dfs))


def balanced_patient_split(df, test_ratio, val_ratio):
    train_list, val_list, test_list = [], [], []

    for center in CENTERS:
        cdf = df[df["center"] == center].copy()
        if len(cdf) == 0:
            continue

        p_df = patient_level_df(cdf)

        patients = p_df["patient"].values
        labels = p_df["label"].values

        if len(np.unique(labels)) < 2 or len(patients) < 5:
            train_list.append(cdf)
            continue

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_ratio,
            random_state=RANDOM_SEED
        )

        tr_idx, te_idx = next(sss.split(patients, labels))

        train_pat = patients[tr_idx]
        test_pat  = patients[te_idx]

        train_labels = labels[tr_idx]

        if len(np.unique(train_labels)) > 1 and len(train_pat) > 4:
            sss_val = StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_ratio,
                random_state=RANDOM_SEED
            )

            tr2_idx, va_idx = next(sss_val.split(train_pat, train_labels))

            final_train = train_pat[tr2_idx]
            val_pat     = train_pat[va_idx]
        else:
            final_train = train_pat
            val_pat = []

        train_list.append(cdf[cdf["patient"].isin(final_train)])
        val_list.append(cdf[cdf["patient"].isin(val_pat)])
        test_list.append(cdf[cdf["patient"].isin(test_pat)])

    return (
        safe_concat(train_list),
        safe_concat(val_list),
        safe_concat(test_list)
    )


def add_cv_folds(train_df):
    if len(train_df) == 0:
        return train_df

    p_df = patient_level_df(train_df)

    if len(p_df) < CV_FOLDS:
        train_df["cv_fold"] = 0
        return train_df

    skf = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    folds = np.zeros(len(p_df), dtype=int)

    for fold, (_, val_idx) in enumerate(skf.split(p_df["patient"], p_df["label"])):
        folds[val_idx] = fold

    p_df["cv_fold"] = folds
    fold_map = dict(zip(p_df["patient"], p_df["cv_fold"]))

    train_df["cv_fold"] = train_df["patient"].map(fold_map)

    return train_df


def print_distribution(df, name):
    if len(df) == 0:
        print(f"{name}: empty")
        return

    total = len(df)
    pos = df["label"].sum()
    neg = total - pos

    print(f"{name}: total={total} | pos={pos} | neg={neg}")


def create_splits():
    labels_csv = Path(DATA_PROC) / "labels.csv"

    if not labels_csv.exists():
        print("Run preprocessing first")
        return

    df = pd.read_csv(labels_csv)

    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print_distribution(df, "ALL")

    print("\n=== SEGMENTATION SPLITS ===")
    tr, va, te = balanced_patient_split(df, TEST_RATIO_SEG, VAL_RATIO)
    tr = add_cv_folds(tr)

    tr.to_csv(f"{SPLITS}/seg_train.csv", index=False)
    va.to_csv(f"{SPLITS}/seg_val.csv", index=False)
    te.to_csv(f"{SPLITS}/seg_test.csv", index=False)

    print_distribution(tr, "Seg Train")
    print_distribution(va, "Seg Val")
    print_distribution(te, "Seg Test")

    print("\n=== CLASSIFICATION SPLITS ===")
    tr, va, te = balanced_patient_split(df, TEST_RATIO_CLS, VAL_RATIO)
    tr = add_cv_folds(tr)

    tr.to_csv(f"{SPLITS}/cls_train.csv", index=False)
    va.to_csv(f"{SPLITS}/cls_val.csv", index=False)
    te.to_csv(f"{SPLITS}/cls_test.csv", index=False)

    print_distribution(tr, "Cls Train")
    print_distribution(va, "Cls Val")
    print_distribution(te, "Cls Test")

    print("\nSplits created successfully")


if __name__ == "__main__":
    create_splits()