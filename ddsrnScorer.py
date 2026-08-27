import torch
from ddsrn import create_ddsrn_model
from extractor import load_feature_extractor, FeatureExtractor
from backbones import Backbone
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------
# Helper function for safe checkpoint loading
# ---------------------------------------------------------

def load_clean_state_dict(model, model_path, device):
    """
    Loads a checkpoint while handling the '_orig_mod.' prefix that torch.compile()
    adds to keys.

    Also gives a clearer error if the checkpoint does not match the current
    DDSRN architecture/backbone.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint at {model_path} does not contain 'model_state_dict'. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    state_dict = checkpoint["model_state_dict"]

    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        clean_state_dict[new_key] = v

    model_state = model.state_dict()

    mismatches = []
    missing_in_checkpoint = []
    unexpected_in_checkpoint = []

    for k, v in clean_state_dict.items():
        if k not in model_state:
            unexpected_in_checkpoint.append(k)
        elif model_state[k].shape != v.shape:
            mismatches.append((k, tuple(v.shape), tuple(model_state[k].shape)))

    for k in model_state:
        if k not in clean_state_dict:
            missing_in_checkpoint.append(k)

    if mismatches or missing_in_checkpoint or unexpected_in_checkpoint:
        print("\nDDSRN checkpoint/model mismatch detected.")
        print(
            "This usually means the checkpoint was trained with a different "
            "Backbone enum/config than the one currently being used."
        )

        if mismatches:
            print("\nShape mismatches:")
            for k, ckpt_shape, model_shape in mismatches[:30]:
                print(f"  {k}")
                print(f"    checkpoint: {ckpt_shape}")
                print(f"    model:      {model_shape}")
            if len(mismatches) > 30:
                print(f"  ... and {len(mismatches) - 30} more shape mismatches")

        if missing_in_checkpoint:
            print("\nKeys missing from checkpoint:")
            for k in missing_in_checkpoint[:30]:
                print(f"  {k}")
            if len(missing_in_checkpoint) > 30:
                print(f"  ... and {len(missing_in_checkpoint) - 30} more missing keys")

        if unexpected_in_checkpoint:
            print("\nUnexpected keys in checkpoint:")
            for k in unexpected_in_checkpoint[:30]:
                print(f"  {k}")
            if len(unexpected_in_checkpoint) > 30:
                print(f"  ... and {len(unexpected_in_checkpoint) - 30} more unexpected keys")

        raise RuntimeError(
            "DDSRN checkpoint does not match the instantiated model. "
            "Use the same Backbone that was used during DDSRN training."
        )

    model.load_state_dict(clean_state_dict)
    return model


# ---------------------------------------------------------
# Image-to-image DDSRN scorer
# ---------------------------------------------------------

class ddsrnScorer(torch.nn.Module):
    def __init__(
        self,
        model_path: str,
        backbone: Backbone,
        weights_path: str = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        self.backbone = backbone
        self.weights_path = weights_path

        self.ddsrn = create_ddsrn_model(
            feature_channels=backbone.config.channels,
            layer_indices=backbone.config.indices,
        ).to(device).eval()

        load_clean_state_dict(self.ddsrn, model_path, device)
        self.ddsrn.eval()

        self.extractor: FeatureExtractor = load_feature_extractor(
            backbone_name=backbone,
            weights_path=weights_path,
        ).to(device).eval()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img1: Tensor [B, C, H, W] or [C, H, W]
            img2: Tensor [B, C, H, W] or [C, H, W]

        Returns:
            Similarity/degradation score. Lower means more similar.
            Shape is scalar or [B], depending on batch size.
        """
        gt_img = self._ensure_batch(img1).to(self.device)
        mod_img = self._ensure_batch(img2).to(self.device)

        gt_feat, mod_feat = self.extractor.extract_features(gt_img, mod_img)

        score = self.ddsrn(gt_feat, mod_feat).squeeze()

        return score

    @staticmethod
    def _ensure_batch(img: torch.Tensor) -> torch.Tensor:
        return img.unsqueeze(0) if img.dim() == 3 else img


def DDSRN(
    ref_img_path,
    deg_img_path,
    model_path,
    backbone=Backbone.YOLO_V11_M,
    weights_path="yolo11m.pt",
    device="cuda",
):
    """
    Compute DDSRN score between a reference and degraded image.

    Example:
        score = DDSRN(
            "clean.jpg",
            "corrupt.jpg",
            model_path="checkpoints/.../best_model.pt",
            backbone=Backbone.YOLO_V11_M,
            weights_path="yolo11m.pt",
            device="cuda"
        )
    """
    metric = ddsrnScorer(
        model_path=model_path,
        backbone=backbone,
        weights_path=weights_path,
        device=device,
    ).eval()

    ref_img = Image.open(ref_img_path).convert("RGB")
    deg_img = Image.open(deg_img_path).convert("RGB")

    ref_tensor = ToTensor()(ref_img)
    deg_tensor = ToTensor()(deg_img)

    with torch.no_grad():
        score = metric(ref_tensor, deg_tensor)

    return score.detach().cpu().item()


# ---------------------------------------------------------
# Feature-dict DDSRN scorer
# ---------------------------------------------------------

class ddsrnFeatScorer(torch.nn.Module):
    def __init__(
        self,
        model_path: str,
        backbone: Backbone,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.backbone = backbone

        self.ddsrn = create_ddsrn_model(
            feature_channels=backbone.config.channels,
            layer_indices=backbone.config.indices,
        ).to(device).eval()

        load_clean_state_dict(self.ddsrn, model_path, device)
        self.ddsrn.eval()

    def _prepare_features(self, feats: dict) -> dict:
        """
        Ensures features are on the correct device and have shape [B, C, H, W].
        """
        processed = {}

        for k, v in feats.items():
            t = v[0] if isinstance(v, list) else v
            t = t.to(self.device)

            if t.dim() == 3:
                t = t.unsqueeze(0)

            processed[k] = t

        return processed

    def forward(self, feats1: dict, feats2: dict) -> torch.Tensor:
        """
        Args:
            feats1: Dict[str, Tensor], features from reference image
            feats2: Dict[str, Tensor], features from degraded image
        """
        keys1 = set(feats1.keys())
        keys2 = set(feats2.keys())

        assert keys1 == keys2, f"Feature keys mismatch: {keys1} vs {keys2}"

        feat1_ready = self._prepare_features(feats1)
        feat2_ready = self._prepare_features(feats2)

        score = self.ddsrn(feat1_ready, feat2_ready).squeeze()

        if score.dim() == 0:
            return score

        return score.mean()


# ---------------------------------------------------------
# Faster R-CNN feature-dict DDSRN scorer
# ---------------------------------------------------------

class ddsrnFeatScorer_FasterRCNN(torch.nn.Module):
    def __init__(
        self,
        model_path: str,
        backbone: Backbone,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.backbone = backbone

        self.ddsrn = create_ddsrn_model(
            feature_channels=backbone.config.channels,
            layer_indices=backbone.config.indices,
        ).to(device).eval()

        load_clean_state_dict(self.ddsrn, model_path, device)
        self.ddsrn.eval()

    def _prepare_features(self, feats: dict) -> dict:
        """
        Ensures features are on the correct device and have shape [B, C, H, W].
        """
        processed = {}

        for k, v in feats.items():
            t = v[0] if isinstance(v, list) else v
            t = t.to(self.device)

            if t.dim() == 3:
                t = t.unsqueeze(0)

            processed[k] = t

        return processed

    def forward(self, feats1: dict, feats2: dict) -> torch.Tensor:
        keys1 = set(feats1.keys())
        keys2 = set(feats2.keys())

        assert keys1 == keys2, f"Feature keys mismatch: {keys1} vs {keys2}"

        feat1_ready = self._prepare_features(feats1)
        feat2_ready = self._prepare_features(feats2)

        score = self.ddsrn(feat1_ready, feat2_ready).squeeze()

        if score.dim() == 0:
            return score

        return score.mean()


# ---------------------------------------------------------
# Convenience function for Faster R-CNN MobileNet V3 FPN
# ---------------------------------------------------------

def DDSRN_FASTER_RCNN_MOBILENET_V3_LARGE_FPN(
    ref_img_path,
    deg_img_path,
    model_path,
    device="cuda",
):
    """
    Compute DDSRN score using the FasterRCNN_MOBILENET_V3_LARGE_FPN backbone.
    """
    backbone = Backbone.FASTERRCNN_MOBILENET_V3_LARGE_FPN

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    ref_img = preprocess(Image.open(ref_img_path).convert("RGB")).unsqueeze(0)
    deg_img = preprocess(Image.open(deg_img_path).convert("RGB")).unsqueeze(0)

    metric = ddsrnScorer(
        model_path=model_path,
        backbone=backbone,
        weights_path=None,
        device=device,
    ).eval()

    with torch.no_grad():
        score = metric(ref_img, deg_img)

    return score.detach().cpu().item()