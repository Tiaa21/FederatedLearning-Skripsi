import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# === CONFIG ===
csv_path = "C:/Users/Chyntia Irawan/Documents/GitHub/curriculum-federated-learning/data/inbreast/index.csv"
img_root = "C:/Users/Chyntia Irawan/Documents/GitHub/curriculum-federated-learning/data/inbreast/images"
out_root = "dataset_inbreast"
n_sites = 3
train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2
random_state = 42

# === LOAD CSV ===
df = pd.read_csv(csv_path)
# pastikan ada kolom 'path' (nama file) dan 'label'
assert "path" in df.columns and "label" in df.columns

# === SPLIT GLOBAL (train/val/test) ===
train_df, test_df = train_test_split(df, test_size=test_ratio, stratify=df["label"], random_state=random_state)
train_df, val_df = train_test_split(train_df, test_size=val_ratio/(train_ratio+val_ratio), stratify=train_df["label"], random_state=random_state)

splits = {"train": train_df, "val": val_df, "test": test_df}

# === BUAT SITE DARI TRAIN SPLIT ===
# bagi train_df jadi n_sites secara stratified
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=n_sites, shuffle=True, random_state=random_state)

site_ids = [None] * len(train_df)
for site, (_, idx) in enumerate(skf.split(train_df["path"], train_df["label"])):
    for i in idx:
        site_ids[i] = site
train_df = train_df.copy()
train_df["site"] = site_ids

# === COPY FILES KE FOLDER ===
for site in range(n_sites):
    for split, split_df in splits.items():
        if split == "train":
            subset = train_df[train_df["site"] == site]
        else:
            subset = split_df  # val & test sama untuk semua site
        for _, row in subset.iterrows():
            p = row["path"]
            if os.path.isabs(p):  # kalau sudah absolute path
                src = p
            else:
                # kalau index.csv simpan path relatif panjang (misal "data/inbreast/images/xxx.png")
                # ambil hanya nama file terakhir
                filename = os.path.basename(p)
                src = os.path.join(img_root, filename)
            dst = os.path.join(out_root, f"site{site}", split, str(row["label"]))
            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, dst)
print("Done! Dataset sudah di-split ke folder:", out_root)