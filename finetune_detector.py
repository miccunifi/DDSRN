from pathlib import Path
import shutil
from ultralytics import YOLO, RTDETR


DATA_YAML = "datasets/Kitti_Visdrone_Dataset/kitti_visdrone.yaml"

EPOCHS = 100
PATIENCE = 20
IMGSZ = 960
WORKERS = 8
DEVICE = 0

CHECKPOINT_DIR = Path("checkpoints/detectors")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. FINE-TUNE YOLO26X
# =============================================================================

print("\n" + "=" * 80)
print("Fine-tuning YOLO26X")
print("=" * 80 + "\n")

yolo = YOLO("yolo26x.pt")

yolo_results = yolo.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    patience=PATIENCE,
    imgsz=IMGSZ,
    batch=-1,
    workers=WORKERS,
    device=DEVICE,

    # Original augmentations
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,

    # Original fine-tuning settings
    lr0=0.01,
    lrf=0.01,

    project="runs/kitti_visdrone",
    name="yolo26ft",
)

yolo_best_src = Path(yolo_results.save_dir) / "weights" / "best.pt"
yolo_best_dst = CHECKPOINT_DIR / "yolo26ft.pt"

shutil.copy2(yolo_best_src, yolo_best_dst)

print(f"[INFO] YOLO26X best checkpoint saved to: {yolo_best_dst}")

# =============================================================================
# 2. FINE-TUNE RT-DETR-L
# =============================================================================

print("\n" + "=" * 80)
print("Fine-tuning RT-DETR-L")
print("=" * 80 + "\n")

rtdetr = RTDETR("rtdetr-l.pt")

rtdetr_results = rtdetr.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    patience=PATIENCE,
    imgsz=IMGSZ,
    batch=-1,
    workers=WORKERS,
    device=DEVICE,

    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,

    lr0=0.01,
    lrf=0.01,

    project="runs/kitti_visdrone",
    name="rtdetr-l_ft",
)

rtdetr_best_src = Path(rtdetr_results.save_dir) / "weights" / "best.pt"
rtdetr_best_dst = CHECKPOINT_DIR / "rtdetr-l_ft.pt"

shutil.copy2(rtdetr_best_src, rtdetr_best_dst)

print(f"[INFO] RT-DETR-L best checkpoint saved to: {rtdetr_best_dst}")