import os, glob, csv
from pathlib import Path
import pandas as pd
import pydicom
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedKFold

# ========== EDIT PATHS ==========
DICOM_DIR = r"C:\Users\Chyntia Irawan\Documents\GitHub\inbreast.tgz\inbreast\ALL-IMGS"
EXCEL = r"C:\Users\Chyntia Irawan\Documents\GitHub\inbreast.tgz\inbreast\INbreast.xls"
OUT_DIR = "data/inbreast"   # hasil hanya di data/inbreast/images + index.csv
# ================================

os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)

# 1) baca excel
df = pd.read_excel(EXCEL, engine='xlrd')
df = df[['Patient ID','File Name','Bi-Rads']].dropna(subset=['File Name'])

def birads_to_binary(x):
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    try:
        v = int(s[0])  # ambil digit pertama (biar "4a", "4b", "4c" tetap kebaca sebagai 4)
        if v <= 2:
            return 0  # normal/benign
        else:
            return 1  # malignant (3–6)
    except:
        return 0

meta = {str(r['File Name']).strip().split('.')[0]: {   # hilangin .dcm kalau ada
            'pid': str(r['Patient ID']).strip(),
            'label': birads_to_binary(r['Bi-Rads'])
        } for _,r in df.iterrows()}

# 2) convert DICOM -> PNG
rows = []
dcm_files = glob.glob(os.path.join(DICOM_DIR, "**", "*.dcm"), recursive=True)
print("Found DICOM files:", len(dcm_files))

for idx, dcm_path in enumerate(dcm_files):
    fname = Path(dcm_path).stem   # buang .dcm
    # ambil hanya 8 digit pertama (ID pasien sesuai Excel)
    key = fname.split("_")[0]  

    if key not in meta:
        continue
    label = meta[key]['label']

    try:
        d = pydicom.dcmread(dcm_path)
        img = d.pixel_array.astype(np.float32)
        img -= img.min(); img /= (img.max() + 1e-8)
        img = (img * 255.0).astype(np.uint8)

        # ===== Resize if too big =====
        H, W = img.shape
        long_side = max(H, W)
        target = 1024   # change to 512 / 2048 depending on your GPU
        scale = target / long_side if long_side > target else 1.0
        newH, newW = int(H * scale), int(W * scale)

        pil = Image.fromarray(img).resize((newW, newH), Image.LANCZOS).convert('RGB')
        # =============================

    except Exception as e:
        print("Failed:", dcm_path, e)
        continue

    outfn = f"{Path(fname).stem}.png"
    outpath = os.path.join(OUT_DIR, "images", outfn)
    pil.save(outpath)

    rows.append([outpath, label, "inbreast", idx])

# 3) Assign pseudo-sites
n_sites = 3  # number of pseudo-sites
labels = np.array([r[1] for r in rows])
skf = StratifiedKFold(n_splits=n_sites, shuffle=True, random_state=42)

for site_idx, (_, idxs) in enumerate(skf.split(rows, labels)):
    for i in idxs:
        rows[i][2] = f"site{site_idx}"

# 4) tulis index.csv
csv_path = os.path.join(OUT_DIR, "index.csv")
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['path','label','domain','idx'])
    for r in rows:
        w.writerow(r)

print("Total samples:", len(rows), "-> index:", csv_path)