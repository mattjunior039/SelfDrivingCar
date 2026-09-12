import os
import shutil
import random
from pathlib import Path
import pandas as pd

# Paths
ROOT = Path("/Users/matthew/Downloads/archive")
OUTPUT = Path.home() / "Desktop" / "lisa_stop_sign_samples"

CLEAN_DIR = OUTPUT / "clean_stop_signs"
SHADOW_OCCLUDED_DIR = OUTPUT / "shadowy_occluded_stop_signs"
DISTRACTOR_DIR = OUTPUT / "distractors"

for d in [CLEAN_DIR, SHADOW_OCCLUDED_DIR, DISTRACTOR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 1. Locate and load CSV with automatic delimiter detection
csv_files = list(ROOT.rglob("allAnnotations.csv")) or list(ROOT.rglob("frameAnnotations.csv"))
if not csv_files:
    raise FileNotFoundError(f"Could not find an annotations CSV file under {ROOT}")

csv_path = csv_files[0]
print(f"Loading annotations from: {csv_path}")

# Peek at the first line to verify delimiter
with open(csv_path, "r", encoding="utf-8-sig") as f:
    first_line = f.readline()
delimiter = ";" if ";" in first_line else ","

df = pd.read_csv(csv_path, sep=delimiter)
df.columns = df.columns.str.strip()
print(f"Detected columns: {df.columns.tolist()}")

# Helper for flexible, case-insensitive column matching
def find_col(candidates):
    for c in df.columns:
        if any(cand.lower() in c.lower() for cand in candidates):
            return c
    return None

fn_col = find_col(["filename", "file"])
tag_col = find_col(["annotation tag", "tag"])
ul_x_col = find_col(["upper left corner x", "ul_x", "x1"])
ul_y_col = find_col(["upper left corner y", "ul_y", "y1"])
lr_x_col = find_col(["lower right corner x", "lr_x", "x2"])
lr_y_col = find_col(["lower right corner y", "lr_y", "y2"])
occ_col = find_col(["occlude"])

# 2. Extract occlusion flags safely
df["Occluded"] = 0
df["On another road"] = 0

if occ_col:
    if "," in str(df[occ_col].iloc[0]):
        meta_split = df[occ_col].astype(str).str.split(",", expand=True)
        df["Occluded"] = pd.to_numeric(meta_split[0].str.strip(), errors="coerce").fillna(0)
        if meta_split.shape[1] > 1:
            df["On another road"] = pd.to_numeric(meta_split[1].str.strip(), errors="coerce").fillna(0)
    else:
        df["Occluded"] = pd.to_numeric(df[occ_col], errors="coerce").fillna(0)

# 3. Compute bounding box area
df["width"] = pd.to_numeric(df[lr_x_col], errors="coerce") - pd.to_numeric(df[ul_x_col], errors="coerce")
df["height"] = pd.to_numeric(df[lr_y_col], errors="coerce") - pd.to_numeric(df[ul_y_col], errors="coerce")
df["area"] = df["width"] * df["height"]

# Filter stop signs
stop_df = df[df[tag_col].astype(str).str.strip().str.lower() == "stop"].copy()
print(f"Total stop sign annotations found: {len(stop_df)}")

# 4. Partition samples
if df["Occluded"].nunique() > 1:
    clean_pool = stop_df[(stop_df["Occluded"] == 0) & (stop_df["On another road"] == 0) & (stop_df["area"] >= 400)]
    shadow_pool = stop_df[(stop_df["Occluded"] == 1) | (stop_df["On another road"] == 1) | (stop_df["area"] < 250)]
else:
    # If occlusion was not annotated, split by prominent vs distant/small signs
    clean_pool = stop_df[stop_df["area"] >= 500]
    shadow_pool = stop_df[stop_df["area"] < 300]

clean_samples = clean_pool.drop_duplicates(subset=[fn_col]).sample(n=min(15, len(clean_pool)), random_state=42)
shadow_samples = shadow_pool.drop_duplicates(subset=[fn_col]).sample(n=min(15, len(shadow_pool)), random_state=42)

# 5. Index images and copy files
print("Indexing image paths in archive...")
image_index = {p.name: p for p in ROOT.rglob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]}

def copy_records(subset, destination):
    count = 0
    for _, row in subset.iterrows():
        fname = Path(row[fn_col]).name
        if fname in image_index:
            shutil.copy2(image_index[fname], destination / fname)
            count += 1
    print(f"Saved {count} frames to: {destination.name}")

copy_records(clean_samples, CLEAN_DIR)
copy_records(shadow_samples, SHADOW_OCCLUDED_DIR)

# 6. Negative distractors
negatives_dirs = list(ROOT.rglob("negatives"))
if negatives_dirs:
    neg_images = [p for p in negatives_dirs[0].glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    selected_negs = random.sample(neg_images, min(15, len(neg_images)))
    for img in selected_negs:
        shutil.copy2(img, DISTRACTOR_DIR / img.name)
    print(f"Saved {len(selected_negs)} negative distractors to: {DISTRACTOR_DIR.name}")
else:
    other_frames = df[df[tag_col].astype(str).str.lower() != "stop"][fn_col].drop_duplicates()
    sampled_other = other_frames.sample(n=min(15, len(other_frames)), random_state=42)
    for fname in sampled_other:
        bname = Path(fname).name
        if bname in image_index:
            shutil.copy2(image_index[bname], DISTRACTOR_DIR / bname)
    print(f"Saved {len(sampled_other)} non-stop distractors to: {DISTRACTOR_DIR.name}")

print(f"\nFinished! Samples are saved to:\n{OUTPUT}")