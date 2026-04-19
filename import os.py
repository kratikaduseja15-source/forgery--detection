import os
import shutil
import csv

# Paths
BASE = r"C:\Users\victus\OneDrive\New folder\forgery project\data\findit2"
TRAIN_TXT = os.path.join(BASE, "train.txt")
TRAIN_IMGS = os.path.join(BASE, "train")

# Output folders
REAL_DIR   = os.path.join(BASE, "sorted", "real")
FORGED_DIR = os.path.join(BASE, "sorted", "forged")

os.makedirs(REAL_DIR,   exist_ok=True)
os.makedirs(FORGED_DIR, exist_ok=True)

real_count = 0
forged_count = 0

with open(TRAIN_TXT, "r") as f:
    lines = f.readlines()

# Skip header line
for line in lines[1:]:
    parts = line.strip().split(",")
    
    if len(parts) < 4:
        continue
    
    filename = parts[0]           # e.g. X00016469622.png
    forged_flag = parts[3]        # 4th column: 1=forged, 0=real

    src = os.path.join(TRAIN_IMGS, filename)

    if not os.path.exists(src):
        continue

    if forged_flag == "1":
        shutil.copy(src, os.path.join(FORGED_DIR, filename))
        forged_count += 1
    else:
        shutil.copy(src, os.path.join(REAL_DIR, filename))
        real_count += 1

print(f"✅ Real images:   {real_count}")
print(f"✅ Forged images: {forged_count}")
print("Done! Check the sorted/ folder")