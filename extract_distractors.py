import random
import shutil
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/matthew/Downloads/archive")
DISTRACTOR_DIR = Path.home() / "Desktop" / "lisa_stop_sign_samples" / "distractors"
DISTRACTOR_DIR.mkdir(parents=True, exist_ok=True)

# 1. Search recursively across all subfolders inside any negatives directories
neg_images = []
for neg_folder in ROOT.rglob("negatives"):
    neg_images.extend([p for p in neg_folder.rglob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

if len(neg_images) >= 15:
    random.seed(42)
    selected = random.sample(neg_images, 15)
    for img in selected:
        shutil.copy2(img, DISTRACTOR_DIR / img.name)
    print(f"Successfully saved {len(selected)} background negatives to: {DISTRACTOR_DIR.name}")
else:
    # 2. Fallback: Frame annotations containing other signs (pedestrian crossing, speed limit, etc.)
    print("No images found in negatives folder; sampling from other road sign frames...")
    csv_file = list(ROOT.rglob("allAnnotations.csv"))[0]
    df = pd.read_csv(csv_file, sep=";")
    df.columns = df.columns.str.strip()
    
    other_signs = df[df["Annotation tag"].astype(str).str.strip().str.lower() != "stop"]
    image_index = {p.name: p for p in ROOT.rglob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]}
    
    sampled_files = other_signs["Filename"].drop_duplicates().sample(n=15, random_state=42)
    copied = 0
    for fn in sampled_files:
        name = Path(fn).name
        if name in image_index:
            shutil.copy2(image_index[name], DISTRACTOR_DIR / name)
            copied += 1
    print(f"Successfully saved {copied} sign-distractor frames to: {DISTRACTOR_DIR.name}")