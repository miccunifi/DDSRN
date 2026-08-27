# DDSRN: Detection Degradation Score Regression Network

Official implementation of:

**Novel Task-Driven Loss to Restore Object Detector Performance under Image Degradation**  
Silvia Dani, Leonardo Galteri, Marco Bertini

DDSRN is a detector-agnostic network designed to estimate degradation relevant to **object detection** and to provide a differentiable task-driven loss for image restoration.

The work introduces:

- **Detection Degradation Score (DDS)**: a full-reference metric measuring how image degradation affects object detections.
- **DDSRN**: a differentiable predictor trained from DDS supervision and usable as a loss without running an object detector during restoration training.

> **Paper:** proceedings link will be added once available.  
> **Pretrained checkpoints:** [Google Drive](https://drive.google.com/drive/folders/1M0GMC2H8H3WqSJavJ2504BRt0QKYhPS8?usp=sharing)

---

## 📏 Detection Degradation Score

DDS compares detector predictions on a reference image and its degraded version.

For a matched detection:

```math
Q_i =
\mathrm{IoU}(D_i^{\mathrm{ref}}, D_i^{\mathrm{deg}})
\cdot
\min\left(
\frac{c_i^{\mathrm{deg}}}{c_i^{\mathrm{ref}}},
1
\right)
```

and

```math
\mathrm{DDS} =
1 -
\frac{\sum_i Q_i}
{\max(N_{\mathrm{ref}}, N_{\mathrm{deg}})}
```

DDS accounts for localization, confidence, classification errors, missed detections, and additional detections.

- **DDS = 0**: no detection degradation.
- **DDS = 1**: maximum degradation.

The implementation is provided in `dds_metric.py`.

---

## 🧠 DDSRN

DDSRN receives a reference/degraded image pair and extracts multi-scale features using a shared encoder. Differences between the two branches are combined to estimate spatial degradation, task-relevant saliency, and a global degradation score.

The network operates at four feature scales with strides 4, 8, 16, and 32.

The main implementation is provided in:

```text
ddsrn_agnostic.py
```

For restoration training, use:

```python
from ddsrn_agnostic import DDSRNFeatureLoss
```

---

## 📁 Repository Structure

```text
DDSRN/
├── dds_metric.py
├── ddsrn_agnostic.py
├── dataloader.py
├── train.py
├── mergeDatasets.py
├── finetune_detector.py
├── requirements.txt
│
├── helpers/
│   ├── 0_corrupt_images.py
│   ├── 1_compute_metrics.py
│   └── 2_compute_correlations.py
│
└── checkpoints/
    ├── detectors/
    │   ├── yolo26ft.pt
    │   └── rtdetr-l_ft.pt
    │
    └── attemptAgnostic_Kitti_Visdrone_FPN_v1/
        └── best_model.pt
```

---

## ⚙️ Installation

The experiments were run with **Python 3.11.14**.

```bash
conda create -n ddsrn python=3.11.14 pip -y
conda activate ddsrn
pip install -r requirements.txt
```

The released requirements use PyTorch 2.9.1 and torchvision 0.24.1 with CUDA 13.0.

---

## 📦 Pretrained Checkpoints

Paper-specific checkpoints are available from:

**[Download checkpoints from Google Drive](https://drive.google.com/drive/folders/1M0GMC2H8H3WqSJavJ2504BRt0QKYhPS8?usp=sharing)**

Place them as follows:

```text
checkpoints/
├── detectors/
│   ├── yolo26ft.pt
│   └── rtdetr-l_ft.pt
│
└── attemptAgnostic_Kitti_Visdrone_FPN_v1/
    └── best_model.pt
```

| Weights | Use |
| --- | --- |
| `yolo26ft.pt` | Fine-tuned YOLO26x used to generate DDSRN supervision |
| `rtdetr-l_ft.pt` | Fine-tuned RT-DETR-L used to generate DDSRN supervision |
| `best_model.pt` | Trained DDSRN used as task-driven loss |

The original `yolo26x.pt` and `rtdetr-l.pt` weights used to initialize the detectors are provided by Ultralytics.

If you only want to **use DDSRN as a loss**, the detector checkpoints are not required.

---

## 🛠️ Using DDSRN as a Loss

Initialize the pretrained loss once:

```python
from ddsrn_agnostic import DDSRNFeatureLoss

ddsrn_loss = DDSRNFeatureLoss(
    model_path="checkpoints/attemptAgnostic_Kitti_Visdrone_FPN_v1/best_model.pt",
    device="cuda",
    loss_weight=1.0,
)
```

Then include it in the restoration objective:

```python
restored = restoration_model(degraded)

loss_rec = reconstruction_loss(restored, reference)
loss_ddsrn = ddsrn_loss(restored, reference)

loss = loss_rec + lambda_ddsrn * loss_ddsrn

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

`restored` and `reference` are expected to have shape `[B, 3, H, W]`, the same spatial resolution, and values in the `[0, 1]` range.

DDSRN remains frozen while gradients propagate through the restored image to the restoration model. No object detector is required at this stage.

---

## 🏋️ Training DDSRN from Scratch

Training requires:

1. building the merged KITTI + VisDrone dataset;
2. fine-tuning the detectors used for DDSRN supervision;
3. training DDSRN.

### 1. Build KITTI + VisDrone

Run:

```bash
python mergeDatasets.py
```

The resulting dataset is stored under:

```text
datasets/Kitti_Visdrone_Dataset/
├── train/
├── val/
└── kitti_visdrone.yaml
```

Check the dataset path in `train.py` before training.

### 2. Fine-tune the Detectors

Run:

```bash
python finetune_detector.py
```

The script fine-tunes YOLO26x and RT-DETR-L on:

```text
datasets/Kitti_Visdrone_Dataset/kitti_visdrone.yaml
```

and stores their best weights as:

```text
checkpoints/detectors/yolo26ft.pt
checkpoints/detectors/rtdetr-l_ft.pt
```

These checkpoints can also be downloaded from the provided Google Drive.

### 3. Train DDSRN

`dataloader.py` dynamically generates clean/degraded image pairs and uses the fine-tuned detector ensemble to provide DDS regression targets and object-saliency supervision.

Run:

```bash
python train.py
```

The best checkpoint is saved as:

```text
checkpoints/attemptAgnostic_Kitti_Visdrone_FPN_v1/best_model.pt
```

---

## 📊 Reproducing the DDS Analysis

The `helpers/` directory contains the pipeline used for the **COCO-C and VOC-C** DDS experiments:

```text
0_corrupt_images.py → 1_compute_metrics.py → 2_compute_correlations.py
```

The experiments use the 15 standard corruptions at severity levels 1–5.

For convenience:

```bash
CORRUPTIONS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
```

### 0. Generate the Corrupted Datasets

`0_corrupt_images.py` crops the clean images, adapts the annotations, and generates the corrupted versions. VOC XML annotations are converted to COCO-compatible JSON.

#### COCO-C

```bash
python helpers/0_corrupt_images.py \
    /path/to/COCO/val2017 \
    COCO_C \
    filename \
    --annotation-file /path/to/COCO/annotations/instances_val2017.json \
    --output-annotation-file COCO_adapted_annotations.json \
    -c $CORRUPTIONS \
    -j 8
```

#### VOC-C

```bash
python helpers/0_corrupt_images.py \
    /path/to/VOC/JPEGImages \
    VOC_C \
    filename \
    --annotation-file /path/to/VOC/Annotations \
    --output-annotation-file VOC_adapted_annotations.json \
    -c $CORRUPTIONS \
    -j 8
```

### 1. Compute mAP, DDS and LPIPS

`1_compute_metrics.py` evaluates the clean and corrupted images and stores mAP, AP50, DDS, and LPIPS for each corruption and severity.

#### COCO-C

```bash
python helpers/1_compute_metrics.py \
    --model yolo26x.pt \
    --clean_dir COCO_C/cropped_images \
    --corrupted_dir COCO_C/corrupted_images \
    --ann_file COCO_adapted_annotations.json \
    --organization filename \
    --compute_perception \
    --output COCO_results_DDS.json \
    --device cuda:0
```

#### VOC-C

```bash
python helpers/1_compute_metrics.py \
    --model yolo26x.pt \
    --clean_dir VOC_C/cropped_images \
    --corrupted_dir VOC_C/corrupted_images \
    --ann_file VOC_adapted_annotations.json \
    --organization filename \
    --compute_perception \
    --output VOC_results_DDS.json \
    --device cuda:0
```

### 2. Compute Correlations and Plots

Set the input and output paths in `helpers/2_compute_correlations.py`:

```python
input_json = "COCO_results_DDS.json"
output_dir = Path("plots/COCO")
```

then run:

```bash
python helpers/2_compute_correlations.py
```

Repeat with the corresponding VOC results to reproduce the VOC-C analysis.

---

## 📚 Citation

The final proceedings citation will be added once available.

<!--```bibtex
@misc{dani_ddsrn,
    title  = {Novel Task-Driven Loss to Restore Object Detector Performance under Image Degradation},
    author = {Dani, Silvia and Galteri, Leonardo and Bertini, Marco}
}
```
-->
---

## 📄 License

See [LICENSE](LICENSE).
