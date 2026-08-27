import argparse
import json
import os
import sys
import numpy as np
import warnings
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# PyTorch / Vision
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Ultralytics
from ultralytics import YOLO

# ---------------------------------------------------------
# OPTIONAL IMPORTS (DDS & LPIPS)
# ---------------------------------------------------------
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dds_metric import match_predictions
except ImportError:
    match_predictions = None
    print("Warning: 'dds_metric.py' not found. DDS scores will be skipped.")

try:
    import lpips
except ImportError:
    lpips = None
    print("Warning: 'lpips' library not installed. LPIPS scores will be skipped.")

# Filter warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. Detection Logic (Optimized with DataLoader)
# ---------------------------------------------------------

class COCOImageDataset(Dataset):
    """Dataset to load images efficiently for Batch Inference"""
    def __init__(self, image_files, annotation_file):
        self.image_files = [Path(p) for p in image_files]
        self.filename_to_id = {}
        
        # Load annotation mapping
        if annotation_file and os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                coco_data = json.load(f)
            # Create map: '000000123.jpg' -> 123
            self.filename_to_id = {img['file_name']: img['id'] for img in coco_data['images']}
        
    def _get_original_filename(self, filepath):
        name = filepath.name
        if name in self.filename_to_id: return name
        
        # Handle corruption naming: 0000123_noise_5.jpg -> 0000123.jpg
        parts = name.split('_')
        if len(parts) > 1:
            potential = parts[0] + filepath.suffix
            if potential in self.filename_to_id: return potential
        return None

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        original_name = self._get_original_filename(image_path)
        
        if not original_name:
            return None # Skip invalid
            
        try:
            # Load PIL Image (YOLOv8 handles resizing/transforms internally)
            img = Image.open(image_path).convert("RGB")
            return img, {'image_id': self.filename_to_id[original_name], 'path': str(image_path)}
        except Exception as e:
            return None

def coco_collate_fn(batch):
    """Custom collate to handle PIL images and filter Nones"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return [], []
    images = [b[0] for b in batch]
    infos = [b[1] for b in batch]
    return images, infos

class COCODetectionGenerator:
    def __init__(self, model_path, device='cuda:0'):
        self.device = device
        # Load Model ONCE
        print(f"Loading YOLO model: {model_path} to {device}...")
        self.model = YOLO(model_path)
        
    def yolo_to_coco_format(self, predictions, infos, score_threshold=0.001):
        coco_results = []
        
        # Map YOLO 0-79 index to COCO 1-90 Category ID
        COCO_MAP = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        ]
        
        for pred, info in zip(predictions, infos):
            if pred.boxes is None: continue
            
            boxes = pred.boxes.xyxy.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
            class_ids = pred.boxes.cls.cpu().numpy().astype(int)
            image_id = info['image_id']
            
            for box, score, class_id in zip(boxes, scores, class_ids):
                if score < score_threshold: continue
                if class_id >= len(COCO_MAP): continue
                
                coco_results.append({
                    'image_id': int(image_id),
                    'category_id': int(COCO_MAP[class_id]),
                    'bbox': [float(box[0]), float(box[1]), float(box[2]-box[0]), float(box[3]-box[1])],
                    'score': float(score)
                })
        return coco_results

    def generate(self, image_files, annotation_file, output_file, batch_size=32, num_workers=4):
        dataset = COCOImageDataset(image_files, annotation_file)
        if len(dataset) == 0:
            return

        # DataLoader handles multi-process loading (CPU) while GPU computes
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            collate_fn=coco_collate_fn,
            pin_memory=True
        )
        
        all_results = []
        
        # Batch Inference Loop
        for images, infos in tqdm(loader, desc="Detection Inference"):
            if not images: continue
            
            # YOLOv8 supports passing a list of PIL images for batched inference
            results = self.model(images, verbose=False, conf=0.001, device=self.device)
            
            batch_coco = self.yolo_to_coco_format(results, infos)
            all_results.extend(batch_coco)
                
        with open(output_file, 'w') as f:
            json.dump(all_results, f)

# ---------------------------------------------------------
# 2. Perception Logic (Sequential - Safer)
# ---------------------------------------------------------

def compute_perception_sequential(target_files, clean_dir, model_path, device='cuda:0'):
    """
    Computes LPIPS and DDS sequentially (one by one) to avoid batching size mismatches.
    Still significantly faster than the original code because models are loaded only once.
    """
    if not target_files: return {}
    
    # 1. Prepare Pairs
    pairs = []
    clean_candidates = {p.name: p for p in Path(clean_dir).glob("*")}
    
    for corrupted in target_files:
        corrupted = Path(corrupted)
        clean_name = None
        
        if corrupted.name in clean_candidates:
            clean_name = corrupted.name
        else:
            parts = corrupted.name.split('_')
            potential = parts[0] + corrupted.suffix
            if potential in clean_candidates:
                clean_name = potential
        
        if clean_name:
            pairs.append((clean_candidates[clean_name], corrupted))
            
    if not pairs: return {}

    # 2. Init Models ONCE (Outside the loop)
    dds_model = None
    lpips_loss = None
    t_lpips = None
    
    if lpips is not None:
        lpips_loss = lpips.LPIPS(net='alex', verbose=False).to(device)
        # Standard LPIPS transform (No resize/padding needed for sequential!)
        t_lpips = transforms.Compose([
            transforms.Resize((256, 256)),  # LPIPS standard size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    if match_predictions is not None and model_path:
        dds_model = YOLO(model_path)

    dds_scores = []
    lpips_scores = []

    # 3. Sequential Loop
    for clean_path, corr_path in tqdm(pairs, desc="Perception (Seq)"):
        try:
            # Load Images
            c_img = Image.open(clean_path).convert("RGB")
            n_img = Image.open(corr_path).convert("RGB")

            # --- LPIPS (Single Pair) ---
            if lpips_loss is not None:
                # Unsqueeze(0) adds the batch dimension: [C, H, W] -> [1, C, H, W]
                c_tensor = t_lpips(c_img).unsqueeze(0).to(device)
                n_tensor = t_lpips(n_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    dist = lpips_loss(c_tensor, n_tensor)
                    lpips_scores.append(dist.item())

            # --- DDS (Single Pair) ---
            if dds_model is not None:
                # YOLO handles resizing internally, so we pass raw PIL images
                c_res = dds_model(c_img, verbose=False, conf=0.25, device=device)[0]
                n_res = dds_model(n_img, verbose=False, conf=0.05, device=device)[0]
                
                matches = match_predictions(c_res, n_res)
                if matches:
                    dds_scores.append(float(matches[0].get("ddscore", 0.0)))
                else:
                    dds_scores.append(0.0)

        except Exception as e:
            # Skip bad images but don't crash
            continue

    return {
        'mean_dds': float(np.mean(dds_scores)) if dds_scores else 0.0,
        'mean_lpips': float(np.mean(lpips_scores)) if lpips_scores else 0.0
    }

# ---------------------------------------------------------
# 3. Main Benchmark Runner
# ---------------------------------------------------------

def run_benchmark(args):
    # Initialize Generator (Loads Model Once)
    generator = COCODetectionGenerator(args.model, args.device)
    final_metrics = {'clean': {}, 'corruptions': {}}
    
    # Pre-fetch clean files
    clean_files = sorted(list(Path(args.clean_dir).glob("*.jpg")) + list(Path(args.clean_dir).glob("*.png")))

    # --- Clean Evaluation ---
    if not args.skip_clean:
        print("\n=== Clean Evaluation ===")
        clean_json = "temp_clean_dets.json"
        
        generator.generate(clean_files, args.ann_file, clean_json, args.batch_size, args.workers)
        
        try:
            coco_gt = COCO(args.ann_file)
            coco_dt = coco_gt.loadRes(clean_json)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            final_metrics['clean'] = {'mAP': coco_eval.stats[0], 'AP50': coco_eval.stats[1]}
        except Exception as e:
            print(f"Clean eval failed: {e}")
            final_metrics['clean'] = {'mAP': 0.0}
            
        if os.path.exists(clean_json): os.remove(clean_json)

    # --- Corruption Evaluation ---
    corruptions = [args.corruption] if args.corruption else [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    for corr in corruptions:
        final_metrics['corruptions'][corr] = {}
        print(f"\n>>> Processing {corr}")
        
        for sev in [1, 2, 3, 4, 5]:
            print(f"Severity {sev}:", end=" ")
            
            # File finding logic
            if args.organization == 'filename':
                target_files = list(Path(args.corrupted_dir).glob(f"*_{corr}_{sev}.*"))
            else:
                target_files = list((Path(args.corrupted_dir) / corr / str(sev)).glob("*"))
                
            if not target_files:
                print("No files found.")
                continue
                
            stats = {}
            
            # 1. mAP Evaluation (Batched)
            temp_json = f"temp_{corr}_{sev}.json"
            generator.generate(target_files, args.ann_file, temp_json, args.batch_size, args.workers)
            
            try:
                # Capture STDOUT to reduce clutter during loop
                # sys.stdout = open(os.devnull, 'w') 
                coco_gt = COCO(args.ann_file)
                coco_dt = coco_gt.loadRes(temp_json)
                evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
                evaluator.evaluate()
                evaluator.accumulate()
                # sys.stdout = sys.__stdout__ # Restore
                evaluator.summarize()
                
                stats['mAP'] = evaluator.stats[0]
                stats['AP50'] = evaluator.stats[1]
                print(f"mAP: {evaluator.stats[0]:.4f}", end=" | ")
            except Exception:
                # sys.stdout = sys.__stdout__
                print("Eval Failed", end=" | ")
                stats['mAP'] = 0.0
            
            if os.path.exists(temp_json): os.remove(temp_json)
            
            # 2. Perception Metrics (Sequential to avoid Stack errors)
            if args.compute_perception:
                p_scores = compute_perception_sequential(
                    target_files, 
                    args.clean_dir, 
                    args.model,
                    device=args.device
                )
                stats.update(p_scores)
                print(f"LPIPS: {stats.get('mean_lpips',0):.4f} DDS: {stats.get('mean_dds',0):.4f}")
            else:
                print("")
                
            final_metrics['corruptions'][corr][sev] = stats

    with open(args.output, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nSaved to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='YOLO model path')
    parser.add_argument('--clean_dir', type=str, required=True)
    parser.add_argument('--corrupted_dir', type=str, required=True)
    parser.add_argument('--ann_file', type=str, required=True)
    parser.add_argument('--output', type=str, default='results.json')
    parser.add_argument('--organization', type=str, default='filename', choices=['filename', 'folder'])
    
    # Performance args
    parser.add_argument('--workers', type=int, default=8, help='CPU workers for loading data')
    parser.add_argument('--batch_size', type=int, default=32, help='GPU Batch size for mAP')
    
    parser.add_argument('--skip_clean', action='store_true')
    parser.add_argument('--compute_perception', action='store_true')
    parser.add_argument('--corruption', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    # Check GPU
    if 'cuda' in args.device and not torch.cuda.is_available():
        print("Warning: CUDA not available, switching to CPU.")
        args.device = 'cpu'

    run_benchmark(args)

    #python helpers/1_compute_metrics.py --model "yolo11m.pt" --clean_dir "COCO_C/cropped_images" --corrupted_dir "COCO_C/corrupted_images" --ann_file "COCO_adapted_annotations.json" --compute_perception --output COCO_results_01.json