import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.ops as ops
import torchvision.transforms.functional as TF
from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from ultralytics import RTDETR, YOLO

from dds_metric import match_predictions


# ==============================================================================
# ENSEMBLE UTILITIES
# ==============================================================================

def fuse_ensemble_predictions(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    iou_threshold: float = 0.65,
) -> torch.Tensor:
    """Fuse predictions from two detectors using class-aware NMS."""

    if boxes_a.numel() == 0 and boxes_b.numel() == 0:
        return torch.empty((0, 6), device=boxes_a.device)

    if boxes_a.numel() == 0:
        return boxes_b

    if boxes_b.numel() == 0:
        return boxes_a

    all_boxes = torch.cat([boxes_a, boxes_b], dim=0)

    coordinates = all_boxes[:, :4]
    scores = all_boxes[:, 4]
    classes = all_boxes[:, 5]

    keep = ops.batched_nms(
        coordinates,
        scores,
        classes,
        iou_threshold,
    )

    return all_boxes[keep]


def create_gaussian_objectness_map(
    image_shape: Tuple[int, int],
    bboxes: List[List[float]],
) -> torch.Tensor:
    """Create an objectness heatmap from detected bounding boxes."""

    height, width = image_shape
    heatmap = torch.zeros((1, height, width), dtype=torch.float32)

    if not bboxes:
        return heatmap

    y = torch.arange(height, dtype=torch.float32).view(height, 1)
    x = torch.arange(width, dtype=torch.float32).view(1, width)

    for x1, y1, x2, y2 in bboxes:
        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            continue

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        sigma_x = max(box_width / 6.0, 1.0)
        sigma_y = max(box_height / 6.0, 1.0)

        gaussian = torch.exp(
            -(
                ((x - center_x) ** 2) / (2.0 * sigma_x**2)
                + ((y - center_y) ** 2) / (2.0 * sigma_y**2)
            )
        )

        heatmap[0] = torch.maximum(heatmap[0], gaussian)

    return heatmap


# ==============================================================================
# DATASET
# ==============================================================================

class DynamicDistortionDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        yolo_path: str,
        detr_path: str,
        mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        seed: int = 42,
        deterministic: bool = False,
    ):
        self.dataset_root = Path(dataset_root)

        self.yolo_path = yolo_path
        self.detr_path = detr_path

        self.mean = mean
        self.std = std
        self.seed = seed
        self.deterministic = deterministic

        self.detector_yolo = None
        self.detector_detr = None

        self.corruption_names = list(get_corruption_names("all"))

        if "super_resolution" not in self.corruption_names:
            self.corruption_names.append("super_resolution")

        self.severities = [1, 2, 3, 4, 5]

        self.n_types = len(self.corruption_names)
        self.n_levels = len(self.severities)
        self.total_variations = self.n_types * self.n_levels + 1

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        self.image_files = sorted(
            path
            for path in self.dataset_root.rglob("*")
            if path.is_file() and path.suffix.lower() in valid_extensions
        )

        if not self.image_files:
            raise RuntimeError(
                f"No images found under dataset root: {self.dataset_root}"
            )

    def _init_detectors(self) -> None:
        """Initialize detector instances lazily inside the worker process."""

        if self.detector_yolo is None:
            self.detector_yolo = YOLO(self.yolo_path).to("cpu")

        if self.detector_detr is None:
            self.detector_detr = RTDETR(self.detr_path).to("cpu")

    def __len__(self) -> int:
        return len(self.image_files)

    @staticmethod
    def apply_super_resolution_distortion(
        image: np.ndarray,
        severity: int,
    ) -> np.ndarray:
        """Simulate resolution degradation followed by bicubic upsampling."""

        settings = {
            1: (2, cv2.INTER_LINEAR),
            2: (3, cv2.INTER_LINEAR),
            3: (4, cv2.INTER_CUBIC),
            4: (4, cv2.INTER_AREA),
            5: (8, cv2.INTER_AREA),
        }

        if severity not in settings:
            return image

        scale, interpolation = settings[severity]

        height, width = image.shape[:2]

        small_height = max(1, height // scale)
        small_width = max(1, width // scale)

        low_resolution = cv2.resize(
            image,
            (small_width, small_height),
            interpolation=interpolation,
        )

        return cv2.resize(
            low_resolution,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )

    @staticmethod
    def _get_randomized_distortion_mask(
        bboxes: List[List[float]],
        width: int,
        height: int,
    ) -> np.ndarray:
        """Sample object- and background-specific distortion regions."""

        distort_background = random.choice([True, False])

        if distort_background:
            mask = np.ones((height, width), dtype=np.uint8)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)

        if not bboxes:
            if not distort_background:
                mask.fill(1)
            return mask

        has_distortion = distort_background

        for x1, y1, x2, y2 in bboxes:
            x1 = int(max(0, x1))
            y1 = int(max(0, y1))
            x2 = int(min(width, x2))
            y2 = int(min(height, y2))

            if x1 >= x2 or y1 >= y2:
                continue

            distort_object = random.choice([True, False])

            if distort_object:
                mask[y1:y2, x1:x2] = 1
                has_distortion = True
            else:
                mask[y1:y2, x1:x2] = 0

        if not has_distortion:
            mask.fill(1)

        return mask

    def __getitem__(self, idx: int) -> Dict:
        self._init_detectors()

        image_path = self.image_files[idx]

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]

        if height < 32 or width < 32:
            image = cv2.resize(
                image,
                (max(width, 32), max(height, 32)),
            )
            height, width = image.shape[:2]

        target_height = max(32, (height // 32) * 32)
        target_width = max(32, (width // 32) * 32)

        start_y = (height - target_height) // 2
        start_x = (width - target_width) // 2

        image = image[
            start_y:start_y + target_height,
            start_x:start_x + target_width,
        ]

        height, width = image.shape[:2]

        # Detector predictions define the spatial objectness supervision.
        with torch.inference_mode():
            result_yolo = self.detector_yolo(
                image,
                verbose=False,
            )

            result_detr = self.detector_detr(
                image,
                verbose=False,
            )

        boxes_yolo = (
            result_yolo[0].boxes.data.cpu()
            if result_yolo[0].boxes is not None
            else torch.empty((0, 6))
        )

        boxes_detr = (
            result_detr[0].boxes.data.cpu()
            if result_detr[0].boxes is not None
            else torch.empty((0, 6))
        )

        fused_boxes = fuse_ensemble_predictions(
            boxes_yolo,
            boxes_detr,
            iou_threshold=0.65,
        )

        detected_bboxes = (
            fused_boxes[:, :4].cpu().tolist()
            if fused_boxes.numel() > 0
            else []
        )

        heatmap = create_gaussian_objectness_map(
            (height, width),
            detected_bboxes,
        )

        if self.deterministic:
            variation_id = (idx * 997) % self.total_variations
        else:
            variation_id = random.randint(
                0,
                self.total_variations - 1,
            )

        if variation_id == 0:
            severity = 0
            distortion_name = "clean"
            distorted_image = image.copy()

        else:
            adjusted_id = variation_id - 1

            type_idx = adjusted_id // self.n_levels
            severity_idx = adjusted_id % self.n_levels

            distortion_name = self.corruption_names[type_idx]
            severity = self.severities[severity_idx]

            if distortion_name == "super_resolution":
                distorted_base = self.apply_super_resolution_distortion(
                    image,
                    severity,
                )
            else:
                distorted_base = corrupt(
                    image,
                    severity=severity,
                    corruption_name=distortion_name,
                )

            mask = self._get_randomized_distortion_mask(
                detected_bboxes,
                width,
                height,
            )

            mask = np.repeat(
                mask[:, :, None],
                3,
                axis=2,
            )

            distorted_image = np.where(
                mask == 1,
                distorted_base,
                image,
            )

        gt_01 = TF.to_tensor(Image.fromarray(image))
        distorted_01 = TF.to_tensor(Image.fromarray(distorted_image))

        gt = TF.normalize(
            gt_01,
            self.mean,
            self.std,
        )

        distorted = TF.normalize(
            distorted_01,
            self.mean,
            self.std,
        )

        return {
            "gt": gt,
            "distorted": distorted,
            "gt_01": gt_01,
            "dist_01": distorted_01,
            "heatmap": heatmap,
            "score": torch.tensor(-1.0, dtype=torch.float32),
            "distortion_name": distortion_name,
            "severity": severity,
            "path": str(image_path),
        }


# ==============================================================================
# DDS SUPERVISION
# ==============================================================================

class BatchDDSWrapper:
    """Compute detector-based DDS targets on the GPU."""

    def __init__(
        self,
        dataloader: DataLoader,
        yolo_path: str,
        detr_path: str,
        device: str = "cuda:0",
    ):
        self.dataloader = dataloader
        self.device = device

        print(f"Initializing DDS detector ensemble on {self.device}")

        self.yolo = YOLO(yolo_path).to(self.device)
        self.detr = RTDETR(detr_path).to(self.device)

        dummy = np.zeros((32, 32, 3), dtype=np.uint8)

        self.yolo.predict(
            dummy,
            device=self.device,
            verbose=False,
            half=True,
        )

        self.detr.predict(
            dummy,
            device=self.device,
            verbose=False,
            half=True,
        )

    def __iter__(self):
        for batch in self.dataloader:
            yield self.process_batch(batch)

    def __len__(self):
        return len(self.dataloader)

    def __getattr__(self, name):
        return getattr(self.dataloader, name)

    def process_batch(self, batch: Dict) -> Dict:
        gt_01 = batch.pop("gt_01")
        dist_01 = batch.pop("dist_01")

        severities = batch["severity"]

        gt_gpu = gt_01.to(
            self.device,
            non_blocking=True,
        )

        distorted_gpu = dist_01.to(
            self.device,
            non_blocking=True,
        )

        with torch.inference_mode():
            yolo_gt = self.yolo.predict(
                gt_gpu,
                verbose=False,
                half=True,
                conf=0.25,
            )

            detr_gt = self.detr.predict(
                gt_gpu,
                verbose=False,
                half=True,
                conf=0.25,
            )

            yolo_distorted = self.yolo.predict(
                distorted_gpu,
                verbose=False,
                half=True,
                conf=0.05,
            )

            detr_distorted = self.detr.predict(
                distorted_gpu,
                verbose=False,
                half=True,
                conf=0.05,
            )

        fused_gt = []
        fused_distorted = []

        for index in range(len(gt_gpu)):
            fused_gt.append(
                fuse_ensemble_predictions(
                    yolo_gt[index].boxes.data,
                    detr_gt[index].boxes.data,
                )
            )

            fused_distorted.append(
                fuse_ensemble_predictions(
                    yolo_distorted[index].boxes.data,
                    detr_distorted[index].boxes.data,
                )
            )

        scores = match_predictions(
            fused_gt,
            fused_distorted,
        )

        final_scores = []

        for severity, score in zip(severities, scores):
            if severity == 0:
                final_scores.append(0.0)
            else:
                final_scores.append(float(score))

        batch["score"] = torch.tensor(
            final_scores,
            dtype=torch.float32,
        )

        return batch


# ==============================================================================
# COLLATE FUNCTION
# ==============================================================================

def collate_dynamic_batch(batch: List[Dict]) -> Dict:
    """Pad variable-resolution samples to the largest image in the batch."""

    gt = [sample["gt"] for sample in batch]
    distorted = [sample["distorted"] for sample in batch]
    gt_01 = [sample["gt_01"] for sample in batch]
    distorted_01 = [sample["dist_01"] for sample in batch]
    heatmaps = [sample["heatmap"] for sample in batch]

    max_height = max(tensor.shape[1] for tensor in gt)
    max_width = max(tensor.shape[2] for tensor in gt)

    def pad_and_stack(tensors):
        padded = []

        for tensor in tensors:
            _, height, width = tensor.shape

            pad_height = max_height - height
            pad_width = max_width - width

            padded.append(
                F.pad(
                    tensor,
                    (0, pad_width, 0, pad_height),
                    value=0,
                )
            )

        return torch.stack(padded)

    return {
        "gt": pad_and_stack(gt),
        "distorted": pad_and_stack(distorted),
        "gt_01": pad_and_stack(gt_01),
        "dist_01": pad_and_stack(distorted_01),
        "heatmap": pad_and_stack(heatmaps),
        "path": [sample["path"] for sample in batch],
        "distortion_name": [
            sample["distortion_name"] for sample in batch
        ],
        "severity": [
            sample["severity"] for sample in batch
        ],
    }


# ==============================================================================
# DATALOADER
# ==============================================================================

def create_dynamic_dataloader(
    dataset_root: str,
    batch_size: int,
    yolo_path: str = "checkpoints/detectors/yolo26ft.pt",
    detr_path: str = "checkpoints/detectors/rtdetr-l_ft.pt",
    num_workers: int = 0,
    seed: int = 42,
    deterministic: bool = False,
    device: str = "cuda:0",
    drop_last: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
):
    dataset = DynamicDistortionDataset(
        dataset_root=dataset_root,
        yolo_path=yolo_path,
        detr_path=detr_path,
        seed=seed,
        deterministic=deterministic,
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": not deterministic,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": drop_last,
        "collate_fn": collate_dynamic_batch,
    }

    # Worker-specific options are valid only when multiprocessing is enabled.
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    raw_loader = DataLoader(**loader_kwargs)

    return BatchDDSWrapper(
        dataloader=raw_loader,
        yolo_path=yolo_path,
        detr_path=detr_path,
        device=device,
    )