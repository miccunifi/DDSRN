import os
import shutil
from pathlib import Path
from ultralytics.data.utils import check_det_dataset as check_dataset

# ==========================================
# 1. DOWNLOAD DATASETS VIA ULTRALYTICS
# ==========================================
print("Checking datasets already downloaded...")
#the scripts download the datasets if not present but two datasets sequentially breaks the second download. So we check if they are already downloaded and if not, download them one by one.
visdrone_info = check_dataset("VisDrone.yaml")
kitti_info = check_dataset("kitti.yaml")

print("\n✅ Datasets checked! Starting the merge process...\n")

# ==========================================
# 2. INTERSECTION MAPPING SETUP
# ==========================================
# Unified Intersection Classes (6 Classes):
# 0: pedestrian, 1: person, 2: bicycle, 3: car, 4: van, 5: truck

# Mapping original VisDrone IDs to the unified IDs
visdrone_map = {
    0: 0,        # pedestrian -> pedestrian
    1: 1,        # people -> person
    2: 2,        # bicycle -> bicycle
    3: 3,        # car -> car
    4: 4,        # van -> van
    5: 5         # truck -> truck
}

# Mapping original KITTI IDs to the unified IDs
kitti_map = {
    3: 0,        # pedestrian -> pedestrian
    4: 1,        # person_sitting -> person
    5: 2,        # cyclist -> bicycle
    0: 3,        # car -> car
    1: 4,        # van -> van
    2: 5         # truck -> truck
}

OUTPUT_DIR = Path(os.getcwd()) / "datasets" / "Kitti_Visdrone_Dataset"

def get_paths(dataset_info, split):
    paths = dataset_info.get(split, [])
    return [paths] if isinstance(paths, str) else paths

datasets_info = [
    {
        "name": "VisDrone",
        "map": visdrone_map,
        "train_imgs": get_paths(visdrone_info, 'train'),
        "val_imgs": get_paths(visdrone_info, 'val'),
    },
    {
        "name": "KITTI",
        "map": kitti_map,
        "train_imgs": get_paths(kitti_info, 'train'),
        "val_imgs": get_paths(kitti_info, 'val'),
    }
]

# ==========================================
# 3. CREATE OUTPUT DIRECTORIES
# ==========================================
for split in ['train', 'val']:
    (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

# ==========================================
# 4. PROCESSING FUNCTION
# ==========================================
def process_directories(img_dirs, split_name, class_map, prefix=""):
    for img_dir in img_dirs:
        img_dir = Path(img_dir)
        lbl_dir = Path(str(img_dir).replace('images', 'labels'))

        if not img_dir.exists():
            continue

        images = list(img_dir.glob("*.*"))
        print(f"Processing {len(images)} images from {img_dir.name} ({prefix})...")

        for img_path in images:
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            new_name = f"{prefix}_{img_path.name}"
            dest_img = OUTPUT_DIR / split_name / "images" / new_name
            dest_lbl = OUTPUT_DIR / split_name / "labels" / f"{prefix}_{img_path.stem}.txt"
            
            shutil.copy(img_path, dest_img)
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            
            with open(dest_lbl, "w") as f_out:
                if lbl_path.exists():
                    with open(lbl_path, "r") as f_in:
                        for line in f_in:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                orig_class = int(parts[0])
                                if orig_class in class_map:
                                    new_class = class_map[orig_class]
                                    new_line = f"{new_class} " + " ".join(parts[1:]) + "\n"
                                    f_out.write(new_line)

# ==========================================
# 5. RUN THE MERGE
# ==========================================
for ds in datasets_info:
    print(f"\n--- Merging {ds['name']} ---")
    process_directories(ds['train_imgs'], 'train', ds['map'], prefix=ds['name'].lower())
    process_directories(ds['val_imgs'], 'val', ds['map'], prefix=ds['name'].lower())

# ==========================================
# 6. GENERATE DATA.YAML
# ==========================================
yaml_content = f"""path: {OUTPUT_DIR.resolve()}
train: train/images
val: val/images

# Intersected Classes
names:
  0: pedestrian
  1: person
  2: bicycle
  3: car
  4: van
  5: truck
"""

yaml_path = OUTPUT_DIR / "kitti_visdrone.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"\n✅ All done! Combined dataset and YAML created at: {OUTPUT_DIR.resolve()}")