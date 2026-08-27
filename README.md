# DDSRN: Detection Degradation Score Regression Network

Official implementation of:

**Novel Task-Driven Loss to Restore Object Detector Performance under Image Degradation**
Silvia Dani, Leonardo Galteri, Marco Bertini

DDSRN is a **detector-agnostic, task-driven image quality model** designed to estimate how image degradation affects downstream object detection. It provides a differentiable surrogate for the proposed **Detection Degradation Score (DDS)** and can be directly used as a loss for image restoration models.

> **Paper:** [link to paper]
> **Pretrained checkpoints:** [Google Drive](GOOGLE_DRIVE_LINK_HERE)

---

## Overview

Image restoration methods are commonly optimized using pixel-wise or perceptual objectives such as L1, PSNR, SSIM, or LPIPS. However, these metrics are designed around signal fidelity or human perception and do not necessarily reflect the behavior of downstream machine vision systems.

A restored image can therefore look perceptually good while still producing significantly worse object detections.

We address this problem through two components:

* **Detection Degradation Score (DDS):** a detector-based full-reference metric that quantifies changes in localization, classification, confidence, false positives, and false negatives between a clean and degraded image.
* **Detection Degradation Score Regression Network (DDSRN):** a detector-agnostic neural network trained to predict detection degradation and provide a differentiable task-aware objective for image restoration.

Once trained, **DDSRN does not require an object detector** and can be directly inserted into an arbitrary differentiable restoration pipeline.

---

# Detection Degradation Score

Let (D_{ref}) and (D_{deg}) be the detections obtained from a reference and degraded image.

Detections are matched according to their class and bounding-box overlap. For each valid match, the quality is determined jointly by localization and confidence:

[
Q_i =
\operatorname{IoU}(D_i^{ref},D_i^{deg})
\cdot
\min
\left(
\frac{c_i^{deg}}{c_i^{ref}},
1
\right),
]

with (Q_i=0) when the predicted classes do not match.

DDS is then defined as

[
DDS =
1 -
\frac{\sum_i Q_i}
{\max(N_{ref},N_{deg})}.
]

The normalization also penalizes missed and additional detections.

Therefore:

* **DDS = 0:** no detection degradation;
* **DDS = 1:** maximum detection degradation.

Unlike mAP, DDS does not require manual ground-truth annotations for each evaluated image. Detector predictions on the clean reference image are used as the reference.

---

# DDSRN

DDS itself relies on discrete detector predictions and matching and is therefore not directly suitable for gradient-based optimization.

DDSRN learns a continuous approximation of detection degradation from image pairs.

## Architecture

DDSRN uses a shared **Multi-Scale Attentive Encoder** for the reference and degraded images.

The encoder combines:

* residual convolutional blocks;
* CBAM channel and spatial attention;
* Group Normalization;
* GELU activations;
* a top-down Feature Pyramid Network.

Four feature levels are extracted:

[
F={F_4,F_8,F_{16},F_{32}},
]

corresponding to strides 4, 8, 16, and 32.

Absolute feature differences between the reference and degraded branches are computed at each scale, aligned to the highest spatial resolution, concatenated, and fused.

Two prediction heads are then applied:

* **Degradation Head:** estimates spatial degradation;
* **Saliency Head:** estimates the task relevance of each spatial region.

The global degradation prediction is obtained through saliency-weighted aggregation.

This encourages DDSRN to prioritize degradations affecting objects and task-relevant structures instead of treating all image regions equally.

---

# Repository Structure

```text
DDSRN/
├── README.md
├── LICENSE
│
├── dds_metric.py
│   └── Detection Degradation Score implementation
│
├── ddsrn_agnostic.py
│   ├── DDSRN architecture
│   └── DDSRNFeatureLoss for restoration training
│
├── dataloader.py
│   └── Dynamic corruption generation and DDS pseudo-target generation
│
├── train.py
│   └── DDSRN training and evaluation
│
├── mergeDatasets.py
│   └── KITTI + VisDrone dataset construction
│
├── finetune_detector.py
│   └── Fine-tuning of YOLO26x and RT-DETR-L used for DDSRN supervision
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

# Installation

Clone the repository:

```bash
git clone https://github.com/miccunifi/DDSRN.git
cd DDSRN
```

The main dependencies used by the repository are:

```bash
pip install \
    torch torchvision \
    ultralytics \
    imagecorruptions \
    opencv-python \
    pillow \
    numpy scipy pandas \
    tqdm wandb \
    pycocotools \
    lpips \
    filetype \
    matplotlib seaborn
```

A CUDA-capable GPU is strongly recommended for detector inference and DDSRN training.

---

# Pretrained Checkpoints

The pretrained checkpoints used in the paper can be downloaded from:

**[Download pretrained checkpoints](GOOGLE_DRIVE_LINK_HERE)**

Place the downloaded files under the repository root using the following structure:

```text
DDSRN/
└── checkpoints/
    ├── detectors/
    │   ├── yolo26ft.pt
    │   └── rtdetr-l_ft.pt
    │
    └── attemptAgnostic_Kitti_Visdrone_FPN_v1/
        └── best_model.pt
```

The checkpoints have different purposes.

| Checkpoint                                                        | Purpose                                                                                               |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `yolo26x.pt`                                                      | Original YOLO26x weights; used for the DDS experiments and as initialization for detector fine-tuning |
| `rtdetr-l.pt`                                                     | Original RT-DETR-L weights used to initialize detector fine-tuning                                    |
| `checkpoints/detectors/yolo26ft.pt`                               | YOLO26x fine-tuned on KITTI+VisDrone for DDSRN pseudo-target generation                               |
| `checkpoints/detectors/rtdetr-l_ft.pt`                            | RT-DETR-L fine-tuned on KITTI+VisDrone for DDSRN pseudo-target generation                             |
| `checkpoints/attemptAgnostic_Kitti_Visdrone_FPN_v1/best_model.pt` | Pretrained DDSRN                                                                                      |

The fine-tuned YOLO and RT-DETR checkpoints are required **only when training DDSRN from scratch**.

If you only want to use the pretrained DDSRN as a loss, you only need:

```text
checkpoints/attemptAgnostic_Kitti_Visdrone_FPN_v1/best_model.pt
```

No detector is required when using the pretrained DDSRN loss.

---

# Using DDSRN as a Loss

The intended interface for using DDSRN in an image restoration pipeline is:

```python
from ddsrn_agnostic import DDSRNFeatureLoss
```

Initialize the frozen pretrained loss once:

```python
from ddsrn_agnostic import DDSRNFeatureLoss

device = "cuda"

ddsrn_loss = DDSRNFeatureLoss(
    model_path="checkpoints/attemptAgnostic_Kitti_Visdrone_FPN_v1/best_model.pt",
    device=device,
    loss_weight=1.0,
)
```

The DDSRN parameters are automatically frozen. Gradients are still propagated through the restored image and therefore back to the restoration network.

During training:

```python
restored = restoration_model(degraded)

loss_reconstruction = reconstruction_loss(restored, reference)
loss_task = ddsrn_loss(restored, reference)

loss = loss_reconstruction + lambda_ddsrn * loss_task

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

`restored` and `reference` should have shape

```text
[B, 3, H, W]
```

and the restored and reference images must have the same spatial resolution.

The repository operates on image tensors in the ([0,1]) range.

### Loss weighting

There are two equivalent ways to control the DDSRN contribution.

You can keep the wrapper weight equal to one:

```python
ddsrn_loss = DDSRNFeatureLoss(
    model_path=checkpoint,
    device=device,
    loss_weight=1.0,
)

loss = loss_reconstruction + lambda_ddsrn * ddsrn_loss(restored, reference)
```

or directly include the weight inside the DDSRN loss:

```python
ddsrn_loss = DDSRNFeatureLoss(
    model_path=checkpoint,
    device=device,
    loss_weight=lambda_ddsrn,
)

loss = loss_reconstruction + ddsrn_loss(restored, reference)
```

Do not apply the same weight in both places.

### Important

The object detectors used to generate DDS supervision during DDSRN training are **not required here**.

A restoration model using the pretrained DDSRN only requires:

```text
restoration model
      │
      ▼
restored image ─┐
                ├── DDSRNFeatureLoss ──► task-driven loss
reference image ┘
```

This makes the loss independent of the downstream detector architecture.

---

# Training DDSRN from Scratch

Training DDSRN requires three stages:

1. build the combined KITTI + VisDrone dataset;
2. fine-tune the YOLO and RT-DETR detectors used to generate training targets;
3. train DDSRN.

---

## 1. Build the KITTI + VisDrone Dataset

Run:

```bash
python mergeDatasets.py
```

The script uses the Ultralytics dataset utilities to obtain KITTI and VisDrone when needed.

The two datasets are merged over their common detection classes:

```text
0: pedestrian
1: person
2: bicycle
3: car
4: van
5: truck
```

The resulting dataset is written to:

```text
datasets/
└── Kitti_Visdrone_Dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    │
    ├── val/
    │   ├── images/
    │   └── labels/
    │
    └── kitti_visdrone.yaml
```

`kitti_visdrone.yaml` is automatically generated and is subsequently used for detector fine-tuning.

---

## 2. Fine-Tune the Supervision Detectors

DDSRN is trained using detector-generated pseudo-ground-truth targets rather than manual annotations.

To reduce dependence on one detector architecture, training supervision is generated using an ensemble composed of:

* **YOLO26x**, a convolutional detector;
* **RT-DETR-L**, a Transformer-based detector.

The repository contains the complete fine-tuning script:

```bash
python finetune_detector.py
```

The script starts from:

```text
yolo26x.pt
rtdetr-l.pt
```

and trains both models on:

```text
datasets/Kitti_Visdrone_Dataset/kitti_visdrone.yaml
```

The provided configuration uses:

```text
epochs:     100
patience:   20
image size: 960
device:     0
batch:      automatic
```

The best checkpoints are automatically copied to:

```text
checkpoints/
└── detectors/
    ├── yolo26ft.pt
    └── rtdetr-l_ft.pt
```

These are the paths expected by the dynamic DDSRN dataloader.

If the pretrained detector checkpoints are downloaded from the provided Google Drive, this fine-tuning step can be skipped.

---

## 3. Dynamic DDSRN Training Data

DDSRN does not use a fixed corrupted dataset during training.

`dataloader.py` dynamically generates degraded/reference pairs and their pseudo-ground-truth targets.

The loader includes:

* image corruptions from `imagecorruptions`;
* severity levels 1–5;
* custom super-resolution degradation;
* clean samples;
* localized object/background corruptions.

The clean image is processed by both fine-tuned detectors:

```text
YOLO26x fine-tuned
        │
        ├──► class-aware NMS ──► clean reference detections
        │
RT-DETR-L fine-tuned
```

The same process is performed for the degraded image.

The resulting detections are compared using DDS to generate the regression target.

The ensemble detections on the clean image are also converted into Gaussian objectness heatmaps to supervise the DDSRN saliency branch.

Object regions and background regions can be independently corrupted during training, forcing DDSRN to learn whether degradation occurs in task-relevant spatial regions rather than simply estimating global image distortion.

---

## 4. Train DDSRN

Before starting training, set the dataset path in `train.py` according to your local setup.

For the dataset generated by `mergeDatasets.py`, use:

```python
DATASET_ROOT = "datasets/Kitti_Visdrone_Dataset"
```

The current training configuration uses:

```text
batch size:          2
learning rate:       2e-4
maximum epochs:      60
early stopping:      15
seed:                42
optimizer:           AdamW
scheduler:           OneCycleLR
mixed precision:     enabled
gradient accumulation: enabled
```

Start training with:

```bash
python train.py
```

The best model according to validation loss is saved as:

```text
checkpoints/attemptAgnostic_Kitti_Visdrone_FPN_v1/best_model.pt
```

Training metrics are logged using Weights & Biases.

Online W&B logging can be disabled in `train.py` with:

```python
USE_ONLINE_WANDB = False
```

---

# Reproducing the DDS Analysis from the Paper

The scripts under `helpers/` reproduce the corruption-based analysis used to validate DDS on **COCO-C** and **VOC-C**.

The workflow is:

```text
original dataset
      │
      ▼
0_corrupt_images.py
      │
      ├── clean cropped images
      ├── corrupted images
      └── adapted COCO annotations
      │
      ▼
1_compute_metrics.py
      │
      ├── mAP
      ├── DDS
      └── LPIPS
      │
      ▼
results.json
      │
      ▼
2_compute_correlations.py
      │
      ├── severity analysis
      ├── DDS ↔ mAP correlation
      ├── LPIPS ↔ mAP correlation
      └── plots
```

The paper evaluates the following 15 corruption types at severity levels 1–5:

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

For exact reproduction, use this set rather than automatically enabling additional corruptions that may be available in newer `imagecorruptions` versions.

---

## Step 0 — Generate COCO-C and VOC-C

The first helper:

```text
helpers/0_corrupt_images.py
```

performs three operations:

1. crops the source images to detector-compatible dimensions;
2. adapts the bounding-box annotations to the crop;
3. creates the corrupted versions for every requested corruption/severity combination.

The script accepts either:

* a COCO-format JSON annotation file; or
* a directory containing PASCAL VOC XML annotations.

VOC annotations are automatically converted to COCO-compatible category IDs.

### Corruption list

For convenience:

```bash
CORRUPTIONS="gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
```

### COCO-C

For example, using COCO images and their corresponding COCO-format annotations:

```bash
python helpers/0_corrupt_images.py \
    /path/to/coco/images \
    COCO_C \
    subdirs \
    --annotation-file /path/to/coco/annotations.json \
    --output-annotation-file COCO_adapted_annotations.json \
    -c $CORRUPTIONS \
    -j 8
```

This produces:

```text
COCO_C/
├── cropped_images/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
│
└── corrupted_images/
    ├── gaussian_noise/
    │   ├── 1/
    │   ├── 2/
    │   ├── 3/
    │   ├── 4/
    │   └── 5/
    │
    ├── shot_noise/
    │   └── ...
    │
    └── ...
```

and:

```text
COCO_adapted_annotations.json
```

which contains the annotations transformed consistently with the cropped clean images.

### VOC-C

For PASCAL VOC, pass the directory containing the XML files instead of a COCO JSON:

```bash
python helpers/0_corrupt_images.py \
    /path/to/VOC/images \
    VOC_C \
    subdirs \
    --annotation-file /path/to/VOC/Annotations \
    --output-annotation-file VOC_adapted_annotations.json \
    -c $CORRUPTIONS \
    -j 8
```

This creates:

```text
VOC_C/
├── cropped_images/
└── corrupted_images/
```

and:

```text
VOC_adapted_annotations.json
```

The VOC XML annotations are converted to the COCO category convention required by the evaluation pipeline.

`0_corrupt_images.py` recursively processes every image under the supplied input directory. If only a particular VOC split is required, the supplied image directory must therefore contain only the images belonging to that split.

### Useful options

Generate only selected severity levels:

```bash
-se 1 2 3
```

Limit the number of images for debugging:

```bash
-n 100
```

Change the number of corruption workers:

```bash
-j 8
```

---

# Step 1 — Compute mAP, DDS, and LPIPS

The second helper is:

```text
helpers/1_compute_metrics.py
```

It evaluates the clean and corrupted datasets and stores the results in a single JSON file.

For each corruption and severity it computes:

* detector mAP;
* AP50;
* mean DDS;
* mean LPIPS.

The detector used for the DDS validation experiments in the paper is **YOLO26x**.

### COCO-C

```bash
python helpers/1_compute_metrics.py \
    --model yolo26x.pt \
    --clean_dir COCO_C/cropped_images \
    --corrupted_dir COCO_C/corrupted_images \
    --ann_file COCO_adapted_annotations.json \
    --organization folder \
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
    --organization folder \
    --compute_perception \
    --output VOC_results_DDS.json \
    --device cuda:0
```

`--organization folder` is required when the corruptions were generated using the `subdirs` option in Step 0.

The resulting JSON follows the structure:

```text
clean
└── mAP, AP50

corruptions
├── gaussian_noise
│   ├── 1
│   │   ├── mAP
│   │   ├── AP50
│   │   ├── mean_dds
│   │   └── mean_lpips
│   ├── 2
│   └── ...
├── shot_noise
└── ...
```

The same detector is used both to evaluate mAP and to compute the detector-based DDS between the clean and corrupted images.

---

# Step 2 — Correlation and Plot Analysis

The final helper is:

```text
helpers/2_compute_correlations.py
```

It reads the JSON generated in Step 1 and performs the analysis used for the DDS validation in the paper.

It computes and visualizes:

* mAP as a function of corruption severity;
* DDS as a function of corruption severity;
* LPIPS as a function of corruption severity;
* Pearson correlation between DDS and mAP;
* Pearson correlation between LPIPS and mAP;
* per-distortion correlations across severity levels;
* aggregate distributions and box plots;
* per-corruption relationship plots;
* a correlation comparison bar chart.

The script currently specifies its input and output paths inside `main()`.

For COCO-C, set:

```python
input_json = "COCO_results_DDS.json"
output_dir = Path("plots/COCO")
```

then run:

```bash
python helpers/2_compute_correlations.py
```

For VOC-C, change the configuration to:

```python
input_json = "VOC_results_DDS.json"
output_dir = Path("plots/VOC")
```

and run again:

```bash
python helpers/2_compute_correlations.py
```

The resulting analysis corresponds to the corruption experiments used in the paper to compare DDS with detector mAP and conventional perceptual quality measures.

---

# DDS Validation Results

DDS was evaluated on COCO-C and VOC-C using 15 corruption types and five severity levels.

Across the evaluated corruption categories, DDS shows a strong monotonic relationship with detector degradation and achieves a Pearson correlation with mAP of at least:

[
r \leq -0.86.
]

In comparison, LPIPS shows substantially weaker alignment with object detection performance; on COCO-C, the best reported correlation with mAP is approximately:

[
r=-0.52.
]

These results indicate that DDS better reflects machine-vision degradation than a conventional human-centric perceptual metric.

---

# DDSRN Regression Results

On the held-out KITTI + VisDrone evaluation split, DDSRN achieves:

| Metric |     DDSRN |
| ------ | --------: |
| PLCC ↑ |  **0.84** |
| SRCC ↑ |  **0.72** |
| MAE ↓  | **0.044** |

---

# DDSRN for Task-Driven Restoration

The paper evaluates DDSRN as an optimization objective for two restoration tasks.

## Super-Resolution

DDSRN is integrated into the SR4IR/SwinIR training framework and evaluated on OD-VIRAT tiny.

At ×4 Super-Resolution:

| Detector     |  Baseline | Best DDSRN |
| ------------ | --------: | ---------: |
| Faster R-CNN |     27.60 |  **28.10** |
| YOLO26x      |     15.23 |  **16.01** |
| RT-DETR-L    | **11.98** |      11.07 |

Values correspond to mAP@0.5:0.95.

The results demonstrate improved downstream performance for Faster R-CNN and YOLO26x while also revealing an architecture-dependent trade-off for RT-DETR-L.

## Low-Light Image Enhancement

DDSRN is also integrated into HVI CIDNet and evaluated on LoLI-Street.

| Method          | mAP@0.5:0.95 |
| --------------- | -----------: |
| Input           |        20.30 |
| HVI CIDNet      |        22.35 |
| DDSRN, λ=1      |        22.18 |
| DDSRN, λ=10     |    **23.01** |
| DDSRN, λ=25     |        22.54 |
| Well-lit oracle |        24.17 |

These experiments demonstrate that a restoration objective explicitly designed around machine vision can improve downstream detection even when conventional perceptual metrics do not indicate the same preference.

---

# Which Files Do I Need?

### I only want to use DDSRN as a loss

Required:

```text
ddsrn_agnostic.py
checkpoints/attemptAgnostic_Kitti_Visdrone_FPN_v1/best_model.pt
```

Use:

```python
from ddsrn_agnostic import DDSRNFeatureLoss
```

No detector is required.

### I want to train DDSRN from scratch

Required:

```text
mergeDatasets.py
finetune_detector.py
dataloader.py
dds_metric.py
ddsrn_agnostic.py
train.py
```

and either download or generate:

```text
checkpoints/detectors/yolo26ft.pt
checkpoints/detectors/rtdetr-l_ft.pt
```

### I want to reproduce the DDS COCO-C / VOC-C experiment

Required:

```text
dds_metric.py
helpers/0_corrupt_images.py
helpers/1_compute_metrics.py
helpers/2_compute_correlations.py
```

together with:

```text
yolo26x.pt
```

and the corresponding COCO/VOC images and annotations.

---

# Citation

If you use DDS or DDSRN in your work, please cite:

```bibtex
@article{dani_ddsrn,
    title  = {Novel Task-Driven Loss to Restore Object Detector Performance under Image Degradation},
    author = {Dani, Silvia and Galteri, Leonardo and Bertini, Marco}
}
```

The citation information will be updated with the final publication details.

---

# Acknowledgements

This work was developed at the University of Florence and Small Pixels.

We thank Jacopo Damerini for his contribution to the early stages of the work.

---

# License

See [LICENSE](LICENSE) for licensing information.
