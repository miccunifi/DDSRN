import torch
from torchvision.ops import box_iou
from typing import List

@torch.no_grad()
def match_predictions(gt_fused: List[torch.Tensor], mod_fused: List[torch.Tensor], iou_threshold: float = 0.05) -> torch.Tensor:
    """
    Calculates ERROR SCORE: 
    0.0 = Perfect (No degradation)
    1.0 = Total Failure (Max degradation)
    """
    batch_errors = []

    for gt_tensor, mod_tensor in zip(gt_fused, mod_fused):
        n_gt = gt_tensor.shape[0]
        n_mod = mod_tensor.shape[0]

        # Case: Both empty (Perfect match, no error)
        if n_gt == 0 and n_mod == 0:
            batch_errors.append(0.0); continue
        
        # Case: One empty (Total mismatch, max error)
        if n_gt == 0 or n_mod == 0:
            batch_errors.append(1.0); continue

        # Extract data from [N, 6] WBF Tensors: (x1, y1, x2, y2, conf, cls)
        gt_boxes, gt_conf, gt_cls = gt_tensor[:, :4], gt_tensor[:, 4], gt_tensor[:, 5]
        mod_boxes, mod_conf, mod_cls = mod_tensor[:, :4], mod_tensor[:, 4], mod_tensor[:, 5]

        # Calculate IoU Matrix [n_mod, n_gt]
        iou_mat = box_iou(mod_boxes, gt_boxes)
        
        # Class Mask: Only same-class matches allowed
        class_mask = (mod_cls.unsqueeze(1) == gt_cls.unsqueeze(0))
        
        # Filter IoU by Class and Threshold
        cost_mat = iou_mat * class_mask
        cost_mat[cost_mat < iou_threshold] = 0.0

        # Greedy matching based on mod_conf (highest confidence first)
        mod_sort = torch.argsort(mod_conf, descending=True)
        cost_mat = cost_mat[mod_sort]
        mod_conf_sorted = mod_conf[mod_sort]

        matched_quality = torch.zeros(n_mod, device=gt_tensor.device)
        gt_used = torch.zeros(n_gt, dtype=torch.bool, device=gt_tensor.device)

        for i in range(n_mod):
            row = cost_mat[i].clone()
            row[gt_used] = 0.0  # Ignore already matched GT boxes
            
            val, idx = torch.max(row, dim=0)
            if val > 0:
                # Quality = IoU * (Conf Ratio)
                c_ratio = torch.clamp(mod_conf_sorted[i] / gt_conf[idx], max=1.0)
                matched_quality[i] = val * c_ratio
                gt_used[idx] = True

        # Detection Quality: 1.0 (Best) -> 0.0 (Worst)
        # Penalizes False Positives and False Negatives via max(n_gt, n_mod)
        avg_quality = matched_quality.sum() / max(n_gt, n_mod)

        # ERROR SCORE: 0.0 (Best) -> 1.0 (Worst)
        batch_errors.append(1.0 - avg_quality.item())

    return torch.tensor(batch_errors)