# DDSRN: Detection Degradation Score Regression Network

Official implementation of:

**Novel Task-Driven Loss to Restore Object Detector Performance under Image Degradation**
Silvia Dani, Leonardo Galteri, Marco Bertini

DDSRN is a detector-agnostic network designed to estimate degradation relevant to **object detection** and to provide a differentiable task-driven loss for image restoration.

The work introduces:

* **Detection Degradation Score (DDS)**: a full-reference metric measuring how image degradation affects object detections.
* **DDSRN**: a differentiable predictor trained from DDS supervision and usable as a loss without running an object detector during restoration training.

> **Paper:** proceedings link will be added after publication.
> **Pretrained checkpoints:** [Google Drive](https://drive.google.com/drive/folders/1M0GMC2H8H3WqSJavJ2504BRt0QKYhPS8?usp=sharing)

---

## Detection Degradation Score

DDS compares detector predictions on a reference image and its degraded version.

For a matched detection:

$$
Q_i =
\operatorname{IoU}(D_i^{ref}, D_i^{deg})
\cdot
\min\left(\frac{c_i^{deg}}{c_i^{ref}},1\right)
$$

and

$$
DDS =
1 -
\frac{\sum_i Q_i}
{\max(N_{ref},N_{deg})}.
$$

DDS therefore accounts for localization, confidence, classification errors, missed detections, and additional detections.

* **DDS = 0**: no detection degradation.
* **DDS = 1**: maximum degradation.

The implementation is provided in `dds_metric.py`.

---

## DDSRN

DDSRN receives a reference/degraded image pair and extracts multi-scale features using a shared encoder. Differences between the two branches are combined to estimate:

* a spatial degradation map;
* task-relevant object saliency;
* a global degradation score.

The network uses four feature scales with strides 4, 8, 16, and 32.

The main implementation is:

```text
ddsrn_agnostic.py
```

For restoration training, the intended public interface is:

```python
from ddsrn_agnostic import DDSRNFeatureLoss
```

---

## Repository Structure

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

## Installation

The experiments were run with **Python 3.11.14**.

```bash
conda create -n ddsrn python=3.11.14 pip -y
conda activate ddsrn

pip install -r requirements.txt
```

The released requirements use PyTorch 2.9.1 and torchvision 0.24.1 with CUDA 13.0.

---

# Pretrained Checkpoints

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

### Weights used

| Weights          | Use                                                     |
| ---------------- | ------------------------------------------------------- |
| `yolo26x.pt`     | YOLO26x initialization and DDS analysis                 |
| `rtdetr-l.pt`    | RT-DETR-L initialization                                |
| `yolo26ft.pt`    | Fine-tuned YOLO26x used to generate DDSRN supervision   |
| `rtdetr-l_ft.pt` | Fine-tuned RT-DETR-L used to generate DDSRN supervision |
| `best_model.pt`  | Trained DDSRN used as task-driven loss                  |

The original `yolo26x.pt` and `rtdetr-l.pt` weights are provided by Ultralytics.

If you only want to **use DDSRN as a loss**, the detector checkpoints are not required.

---

# Using DDSRN as a Loss

Initialize the pretrained loss once:

```python
from ddsrn_agnostic import DDSRNFeatureLoss

ddsrn_loss = DDSRNFeatureLoss(
    model_path="checkpoints/attemptAgnostic_Kitti_Visdrone_FPN_v1/best_model.pt",
    device="cuda",
    loss_weight=1.0,
)
```

Then add it to the restoration objective:

```python
restored = restoration_model(degraded)

loss_rec = reconstruction_loss(restored, reference)
loss_ddsrn = ddsrn_loss(restored, reference)

loss = loss_rec + lambda_ddsrn * loss_ddsrn

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

`restored` and `reference` are expected as tensors with shape

```text
[B, 3, H, W]
```

and values in the `[0, 1]` range.

DDSRN is kept frozen, while gradients propagate through the restored image to the restoration model.

No object detector is required at this stage.

---

# Training DDSRN from Scratch

Training requires:

1. the merged KITTI + VisDrone dataset;
2. fine-tuned YOLO26x and RT-DETR-L checkpoints;
3. DDSRN training.

## 1. Build KITTI + VisDrone

Run:

```bash
python mergeDatasets.py
```

The script prepares a joint KITTI + VisDrone detection dataset using their common classes and generates:

```text
datasets/Kitti_Visdrone_Dataset/
├── train/
├── val/
└── kitti_visdrone.yaml
```

Check the dataset path in `train.py` before training.

---

## 2. Fine-tune the Detectors

The detectors used to generate DDSRN supervision can be reproduced with:

```bash
python finetune_detector.py
```

The script starts from:

```text
yolo26x.pt
rtdetr-l.pt
```

and fine-tunes both models on:

```text
datasets/Kitti_Visdrone_Dataset/kitti_visdrone.yaml
```

The best weights are copied automatically to:

```text
checkpoints/detectors/yolo26ft.pt
checkpoints/detectors/rtdetr-l_ft.pt
```

These checkpoints are also available from the provided Google Drive, so this step can be skipped when using the released weights.

---

## 3. Train DDSRN

`dataloader.py` dynamically generates clean/degraded image pairs during training and uses the fine-tuned detector ensemble to generate:

* DDS regression targets;
* object saliency supervision.

Training corruptions include the `imagecorruptions` distortions, multiple severity levels, clean samples, super-resolution degradation, and localized object/background degradation.

Run:

```bash
python train.py
```

The best DDSRN model is saved under:

```text
checkpoints/attemptAgnostic_Kitti_Visdrone_FPN_v1/best_model.pt
```

---

# Reproducing the DDS Analysis

The `helpers/` directory contains the pipeline used for the **COCO-C and VOC-C DDS experiments**:

```text
0_corrupt_images.py
        ↓
1_compute_metrics.py
        ↓
2_compute_correlations.py
```

The experiments use the 15 standard corruptions:

```text
gaussian_noise
shot_noise
impulse_noise
defocus_blur
glass_blur
motion_blur
zoom_blur
snow
frost
fog
brightness
contrast
elastic_transform
pixelate
jpeg_compression
```

with severity levels **1–5**.

---

## 0. Generate the Corrupted Datasets

`helpers/0_corrupt_images.py`:

* crops the clean images;
* adapts their bounding boxes;
* generates corrupted images;
* converts VOC XML annotations to COCO-compatible JSON when necessary.

### COCO-C

```bash
python helpers/0_corrupt_images.py \
    /path/to/COCO/val2017 \
    COCO_C \
    filename \
    --annotation-file /path/to/COCO/annotations/instances_val2017.json \
    --output-annotation-file COCO_adapted_annotations.json \
    -c gaussian_noise shot_noise impulse_noise \
       defocus_blur glass_blur motion_blur zoom_blur \
       snow frost fog brightness contrast elastic_transform \
       pixelate jpeg_compression \
    -j 8
```

Output:

```text
COCO_C/
├── cropped_images/
└── corrupted_images/

COCO_adapted_annotations.json
```

### VOC-C

For VOC, provide the image directory and the directory containing the XML annotations:

```bash
python helpers/0_corrupt_images.py \
    /path/to/VOC/JPEGImages \
    VOC_C \
    filename \
    --annotation-file /path/to/VOC/Annotations \
    --output-annotation-file VOC_adapted_annotations.json \
    -c gaussian_noise shot_noise impulse_noise \
       defocus_blur glass_blur motion_blur zoom_blur \
       snow frost fog brightness contrast elastic_transform \
       pixelate jpeg_compression \
    -j 8
```

Output:

```text
VOC_C/
├── cropped_images/
└── corrupted_images/

VOC_adapted_annotations.json
```

---

## 1. Compute mAP, DDS and LPIPS

`helpers/1_compute_metrics.py` evaluates the detector on the clean and corrupted images and computes:

* mAP;
* AP50;
* DDS;
* LPIPS.

### COCO-C

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

### VOC-C

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

The resulting JSON contains the clean detector performance and the results for every corruption/severity pair.

---

## 2. Compute Correlations and Generate Plots

`helpers/2_compute_correlations.py` reads the JSON generated in the previous step and produces:

* DDS–mAP correlations;
* LPIPS–mAP correlations;
* per-corruption correlations;
* severity plots;
* comparison plots used for the paper analysis.

Set the input and output paths near the bottom of the script:

```python
input_json = "COCO_results_DDS.json"
output_dir = Path("plots/COCO")
```

then run:

```bash
python helpers/2_compute_correlations.py
```

Repeat with the VOC results:

```python
input_json = "VOC_results_DDS.json"
output_dir = Path("plots/VOC")
```

---

# Main Results

DDS shows a strong relationship with object detection degradation on COCO-C and VOC-C, reaching a correlation with mAP of at least **−0.86** across the evaluated corruption categories.

DDSRN predicts DDS on held-out KITTI + VisDrone data with:

| Metric |    Result |
| ------ | --------: |
| PLCC ↑ |  **0.84** |
| SRCC ↑ |  **0.72** |
| MAE ↓  | **0.044** |

The paper further evaluates DDSRN as a restoration objective for super-resolution and low-light image enhancement.

Full experimental results and ablations are reported in the paper.

---

# Citation

The final citation and proceedings link will be added after publication.

```bibtex
@misc{dani_ddsrn,
    title  = {Novel Task-Driven Loss to Restore Object Detector Performance under Image Degradation},
    author = {Dani, Silvia and Galteri, Leonardo and Bertini, Marco}
}
```

---

# License

See [LICENSE](LICENSE).
