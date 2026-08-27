import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. CBAM ATTENTION BLOCKS (Standard Component)
# ==============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced_dim = max(in_planes // ratio, 4)
        self.fc1 = nn.Conv2d(in_planes, reduced_dim, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_dim, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))

class CBAMResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, attention_kernel=7):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention(kernel_size=attention_kernel)
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        res = self.shortcut(x)
        x = F.gelu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        x = self.ca(x) * x
        x = self.sa(x) * x
        return F.gelu(x + res)

# ==============================================================================
# 2. MULTI-SCALE ATTENTIVE ENCODER (4-Level FPN Core)
# ==============================================================================
class MultiScaleAttentiveEncoder(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU()
        )
        # Backbone Extractions
        self.enc1 = CBAMResBlock(64, 64, 3)    # Stride 4
        self.down1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc2 = CBAMResBlock(128, 128, 5)  # Stride 8
        self.down2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.enc3 = CBAMResBlock(256, 256, 7)  # Stride 16
        self.down3 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.enc4 = CBAMResBlock(512, 512, 7)  # Stride 32

        # FPN Top-Down Pathway
        self.reduce_4 = nn.Conv2d(512, 256, 1)
        self.reduce_3 = nn.Conv2d(256, 128, 1)
        self.reduce_2 = nn.Conv2d(128, 64, 1)
        
        # Smoothing & Standardization (All levels output 'out_channels' = 128)
        self.smooth_s32 = CBAMResBlock(512, out_channels, 3) # Large COCO objects
        self.smooth_s16 = CBAMResBlock(256, out_channels, 3) # Medium COCO objects
        self.smooth_s8  = CBAMResBlock(128, out_channels, 3) # Small VisDrone objects
        self.smooth_s4  = CBAMResBlock(64, out_channels, 3)  # Tiny VisDrone objects

    def forward(self, x):
        # Bottom-Up
        c1 = self.enc1(self.stem(x))
        c2 = self.enc2(self.down1(c1))
        c3 = self.enc3(self.down2(c2))
        c4 = self.enc4(self.down3(c3))

        # Top-Down FPN
        p3 = c3 + F.interpolate(self.reduce_4(c4), size=c3.shape[2:], mode='bilinear')
        p2 = c2 + F.interpolate(self.reduce_3(p3), size=c2.shape[2:], mode='bilinear')
        p1 = c1 + F.interpolate(self.reduce_2(p2), size=c1.shape[2:], mode='bilinear')

        # Dictionary of all 4 scales
        return {
            "stride4":  self.smooth_s4(p1),  
            "stride8":  self.smooth_s8(p2),  
            "stride16": self.smooth_s16(p3), 
            "stride32": self.smooth_s32(c4)  
        }

# ==============================================================================
# 3. MAIN SCORER (4-Level Weighted Aggregation)
# ==============================================================================
class BackboneAgnosticDDSRN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MultiScaleAttentiveEncoder(out_channels=128)
        
        # 4 levels * 128 channels = 512 total
        self.fusion = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.GroupNorm(16, 256), nn.GELU()
        )

        self.obj_head = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.GELU(),
                                      nn.Conv2d(64, 1, 1))
        
        self.deg_head = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.GELU(),
                                      nn.Conv2d(64, 1, 1), nn.Sigmoid())

    def forward(self, gt_img, mod_img, return_all=False):
        f_gt = self.encoder(gt_img)
        f_mod = self.encoder(mod_img)

        # Compute abs differences at all 4 scales and align them to Stride 4
        target_size = f_gt['stride4'].shape[2:]
        d4  = torch.abs(f_gt['stride4'] - f_mod['stride4'])
        d8  = F.interpolate(torch.abs(f_gt['stride8'] - f_mod['stride8']), size=target_size, mode='bilinear')
        d16 = F.interpolate(torch.abs(f_gt['stride16'] - f_mod['stride16']), size=target_size, mode='bilinear')
        d32 = F.interpolate(torch.abs(f_gt['stride32'] - f_mod['stride32']), size=target_size, mode='bilinear')

        feat = self.fusion(torch.cat([d4, d8, d16, d32], dim=1))
        # --- THE FIX ---
        obj_logits = self.obj_head(feat)             # Raw logits for the Loss Function
        obj_probs = torch.sigmoid(obj_logits)        # Probabilities for the Global Score
        deg_map = self.deg_head(feat)

        # Global Score (Using probabilities!)
        num = torch.sum(deg_map * obj_probs, dim=[1, 2, 3])
        den = torch.sum(obj_probs, dim=[1, 2, 3]) + 1e-7
        global_score = num / den

        if return_all:
            return global_score, deg_map, obj_logits, f_gt, f_mod
        return global_score, deg_map

def create_agnostic_model():
    return BackboneAgnosticDDSRN()
# ==============================================================================
# 4. RESTORATION LOSS WRAPPER (Applying Loss to All Levels)
# ==============================================================================
class DDSRNFeatureLoss(nn.Module):
    def __init__(self, model_path=None, device='cuda', loss_weight=1.0):
        super().__init__()
        self.device = device
        self.loss_weight = loss_weight
        self.net = BackboneAgnosticDDSRN().to(device).eval()
        if model_path:
            try:
                print(f"Loading DDSRN Feature Extractor from {model_path}...")               
                # Robust State Dict Loading
                self.load_clean_state_dict(self.net, model_path, device)
                print("Weights loaded successfully.")
            except Exception as e:
                print(f"[WARNING] Failed to load DDSRN weights: {e}")
                print("[WARNING] Metrics will be random!")
        else:
            print("[WARNING] No model_path provided. Using random weights!")
            
        self.net.to(device)
        self.net.eval()
        
        # Freeze Gradients
        for param in self.net.parameters():
            param.requires_grad = False

    def forwardv6(self, sr, hr):
        sr, hr = self._prepare(sr), self._prepare(hr)
        _, deg_map, obj_pred, f_hr, f_sr = self.net(hr, sr, return_all=True)
        
        heatmap = obj_pred.detach()

        # 1. Dynamic Mask Alignment
        mask = F.interpolate(heatmap, size=deg_map.shape[2:], mode='bilinear')

        # 2. Context-Aware Masking
        soft_mask = torch.clamp(mask, min=0.0) 
        combined_weight = deg_map.detach() * soft_mask * 2.0 

        total_loss = 0.0
        
        # 3. Pure Edge-Level Iteration (Extreme Decoupling)
        # We heavily target stride4 for crisp bounding box edges, 
        # but brutally cut off the mid/deep layers to stop texture artifacting.
        level_weights = {
            'stride4':  1.0,  # 100% focus: crisp boundaries
            'stride8':  0.1,  # 10% focus: bare minimum structural support
            'stride16': 0.0,  # ZERO: Stop hallucinating texture noise
            'stride32': 0.0   # ZERO: Stop hallucinating semantic noise
        }
        
        for level, depth_weight in level_weights.items():
            if depth_weight == 0.0:
                continue # Skip completely
                
            # --- THE V6 FIX: L2 NORMALIZATION ---
            # This is the "Universal Translator" from v2. It forces the network 
            # to match the SHAPE of the edges, but strictly prevents the massive 
            # magnitude spikes that cause high-frequency fuzz.
            sr_norm = F.normalize(f_sr[level], p=2, dim=1)
            hr_norm = F.normalize(f_hr[level], p=2, dim=1)
                
            m = F.interpolate(combined_weight, size=f_sr[level].shape[2:], mode='bilinear')
            
            # Smooth L1 (Huber) Loss on the NORMALIZED features
            base_loss = F.smooth_l1_loss(sr_norm, hr_norm, reduction='none', beta=0.1)
            
            # --- THE YOLO/DETR BACKGROUND FIX ---
            # Multiply by dynamic mask AND the depth weight.
            # NOTE: Because it is (1.0 + m), the background mask is 0, meaning 
            # the background penalty is exactly 1.0 * base_loss. 
            # Objects get (1.0 + extra_weight). This perfectly preserves the 
            # natural background context for YOLO/RT-DETR while hyper-focusing on cars.
            total_loss += torch.mean(base_loss * (1.0 + m)) * depth_weight
            
        return total_loss * self.loss_weight

    def forwardv5(self, sr, hr):
        sr, hr = self._prepare(sr), self._prepare(hr)
        _, deg_map, obj_pred, f_hr, f_sr = self.net(hr, sr, return_all=True)
        
        heatmap = obj_pred.detach()

        # 1. Dynamic Mask Alignment
        mask = F.interpolate(heatmap, size=deg_map.shape[2:], mode='bilinear')

        # 2. ZERO Background Floor (The YOLO/DETR Fix)
        # We remove the 0.1 floor. We only want to sharpen the actual objects.
        # This keeps the background perfectly clean, preserving YOLO's global context 
        # and lowering your overall LPIPS back to the baseline.
        soft_mask = torch.clamp(mask, min=0.0) 
        combined_weight = deg_map.detach() * soft_mask * 2.0 

        total_loss = 0.0
        
        # 3. Pure Edge-Level Iteration (Extreme Decoupling)
        # We heavily target stride4 for crisp bounding box edges, 
        # but brutally cut off the mid/deep layers to stop texture artifacting.
        level_weights = {
            'stride4':  1.0,  # 100% focus: crisp boundaries
            'stride8':  0.1,  # 10% focus: bare minimum structural support
            'stride16': 0.0,  # ZERO: Stop hallucinating texture noise
            'stride32': 0.0   # ZERO: Stop hallucinating semantic noise
        }
        
        for level, depth_weight in level_weights.items():
            if depth_weight == 0.0:
                continue # Skip completely
                
            m = F.interpolate(combined_weight, size=f_sr[level].shape[2:], mode='bilinear')
            
            # Smooth L1 (Huber) Loss on raw features
            base_loss = F.smooth_l1_loss(f_sr[level], f_hr[level], reduction='none', beta=0.1)
            
            # Multiply by dynamic mask AND the depth weight
            # If the mask is 0 (background), the loss penalty is perfectly 0.
            total_loss += torch.mean(base_loss * (1.0 + m)) * depth_weight
            
        return total_loss * self.loss_weight
    
    def forwardv4(self, sr, hr):
        sr, hr = self._prepare(sr), self._prepare(hr)
        _, deg_map, obj_pred, f_hr, f_sr = self.net(hr, sr, return_all=True)
        
        heatmap = obj_pred.detach()

        # # 1. Dynamic Mask Alignment
        mask = F.interpolate(heatmap, size=deg_map.shape[2:], mode='bilinear')

        # 2. Universal Floor & Contrast
        # A 0.1 floor preserves enough background for KITTI's context (roads, lanes),
        # while multiplying the object regions by 2.0 gives them strict priority.
        soft_mask = torch.clamp(mask, min=0.1)
        combined_weight = deg_map.detach() * soft_mask * 2.0 

        total_loss = 0.0
        
        # 3. Dynamic Inverse-Stride Weighting (Universal & Plug-and-Play)
        # We tie the loss weight directly to the network's spatial stride.
        # This naturally prioritizes high-frequency edges (SR's actual job) 
        # and suppresses deep semantic hallucination, requiring zero tuning.
        loss_levels = {'stride4': 4, 'stride8': 8, 'stride16': 16, 'stride32': 32}
        base_stride = 4.0 
        
        for level_name, stride_val in loss_levels.items():
            # Automatically calculates: 1.0, 0.5, 0.25, 0.125
            depth_weight = base_stride / stride_val 
            
            # Align the attention mask to the current spatial resolution
            m = F.interpolate(combined_weight, size=f_sr[level_name].shape[2:], mode='bilinear')
            
            # Smooth L1 (Huber) Loss on raw features
            base_loss = F.smooth_l1_loss(f_sr[level_name], f_hr[level_name], reduction='none', beta=0.1)
            
            # Apply BOTH the dynamic spatial mask AND the automatic depth weight
            total_loss += torch.mean(base_loss * (1.0 + m)) * depth_weight
            
        return total_loss * self.loss_weight

    def forwardv3(self, sr, hr):
        sr, hr = self._prepare(sr), self._prepare(hr)
        _, deg_map, obj_pred, f_hr, f_sr = self.net(hr, sr, return_all=True)
        
        heatmap = obj_pred.detach()

        # # 1. Dynamic Mask Alignment
        mask = F.interpolate(heatmap, size=deg_map.shape[2:], mode='bilinear')

        # 2. Universal Floor & Contrast
        # A 0.1 floor preserves enough background for KITTI's context (roads, lanes),
        # while multiplying the object regions by 2.0 gives them strict priority.
        soft_mask = torch.clamp(mask, min=0.1)
        combined_weight = deg_map.detach() * soft_mask * 2.0 

        total_loss = 0.0
        
        # 3. Uniform Scale Iteration (No Hardcoded Biases)
        # We treat all strides equally. The `combined_weight` mask will naturally 
        # activate the correct stride based on the dataset's object sizes.
        loss_levels = ['stride4', 'stride8', 'stride16', 'stride32']
        
        for level in loss_levels:
            # REMOVE the L2 Normalization. Let the features keep their natural magnitude.
            # sr_norm = F.normalize(f_sr[level], p=2, dim=1)  <-- Delete
            # hr_norm = F.normalize(f_hr[level], p=2, dim=1)  <-- Delete
            
            # Align the attention mask to the current spatial resolution
            m = F.interpolate(combined_weight, size=f_sr[level].shape[2:], mode='bilinear')
            
            # 4. Smooth L1 (Huber) Loss directly on the raw features.
            # This alone prevents gradient explosions without destroying magnitude.
            base_loss = F.smooth_l1_loss(f_sr[level], f_hr[level], reduction='none', beta=0.1)
            
            # Apply your dynamic mask and average
            total_loss += torch.mean(base_loss * (1.0 + m))
            
        return total_loss * self.loss_weight
    
    def forward(self, sr, hr): #v2
        sr, hr = self._prepare(sr), self._prepare(hr)
        _, deg_map, obj_pred, f_hr, f_sr = self.net(hr, sr, return_all=True)
        
        heatmap = obj_pred.detach()

        # # 1. Dynamic Mask Alignment
        mask = F.interpolate(heatmap, size=deg_map.shape[2:], mode='bilinear')

        # 2. Universal Floor & Contrast

        soft_mask = torch.clamp(mask, min=0.1)
        combined_weight = deg_map.detach() * soft_mask * 2.0 

        total_loss = 0.0
        
        # 3. Uniform Scale Iteration (No Hardcoded Biases)
        # We treat all strides equally. The `combined_weight` mask will naturally 
        # activate the correct stride based on the dataset's object sizes.
        loss_levels = ['stride4', 'stride8', 'stride16', 'stride32']
        
        for level in loss_levels:
            # 1. L2 Normalization (The YOLO Fix)
            # This explicitly strips away the raw magnitude and forces the network 
            # to only match the structural direction/shape of the features.
            sr_norm = F.normalize(f_sr[level], p=2, dim=1)
            hr_norm = F.normalize(f_hr[level], p=2, dim=1)
            
            # Align the attention mask to the current spatial resolution
            m = F.interpolate(combined_weight, size=sr_norm.shape[2:], mode='bilinear')
            
            # 2. Smooth L1 (Huber) Loss on the NORMALIZED features.
            # Calculates the structural difference safely capped between -1 and 1.
            base_loss = F.smooth_l1_loss(sr_norm, hr_norm, reduction='none', beta=0.1)
            
            # Apply your dynamic mask and average
            total_loss += torch.mean(base_loss * (1.0 + m))
            
        return total_loss * self.loss_weight

    def forwardv7(self, sr, hr):
        sr, hr = self._prepare(sr), self._prepare(hr)
        _, deg_map, obj_pred, f_hr, f_sr = self.net(hr, sr, return_all=True)
        
        heatmap = obj_pred.detach()
        mask = F.interpolate(heatmap, size=deg_map.shape[2:], mode='bilinear')
        soft_mask = torch.clamp(mask, min=0.1)
        combined_weight = deg_map.detach() * soft_mask * 2.0

        total_loss = 0.0

        # ── STRIDE 4: bordi → raw features, massima priorità ──────────────────
        # NON normalizzare: la magnitudine dei bordi è il segnale.
        # Smooth L1 con beta basso = sensibile ai piccoli errori di edge.
        m4 = F.interpolate(combined_weight, size=f_sr['stride4'].shape[2:], mode='bilinear')
        loss_s4 = F.smooth_l1_loss(f_sr['stride4'], f_hr['stride4'], reduction='none', beta=0.05)
        total_loss += torch.mean(loss_s4 * (1.0 + m4)) * 1.0

        # ── STRIDE 8: struttura fine → instance norm, peso medio ──────────────
        # Instance norm: rimuove offset assoluto ma preserva differenze relative.
        # Meglio di L2 norm perché mantiene il contrasto interno del feature map.
        m8 = F.interpolate(combined_weight, size=f_sr['stride8'].shape[2:], mode='bilinear')
        s8_sr = self._instance_norm(f_sr['stride8'])
        s8_hr = self._instance_norm(f_hr['stride8'])
        loss_s8 = F.smooth_l1_loss(s8_sr, s8_hr, reduction='none', beta=0.1)
        total_loss += torch.mean(loss_s8 * (1.0 + m8)) * 0.4

        # ── STRIDE 16: semantica media → cosine loss, peso basso ──────────────
        # Cosine similarity loss: vuoi che il vettore semantico "punti" nella 
        # stessa direzione. La magnitudine assoluta qui non conta, 
        # conta che "somigli ad un'auto" e non ad uno sfondo.
        m16 = F.interpolate(combined_weight, size=f_sr['stride16'].shape[2:], mode='bilinear')
        cos_sim = F.cosine_similarity(f_sr['stride16'], f_hr['stride16'], dim=1, eps=1e-8)
        loss_s16 = (1.0 - cos_sim).unsqueeze(1)  # [B,1,H,W]
        total_loss += torch.mean(loss_s16 * (1.0 + m16)) * 0.15

        # ── STRIDE 32: semantica globale → cosine loss, peso minimo ───────────
        # Mantieni vivo il segnale semantico profondo, ma non lasciare che 
        # domini. Senza questo, il detector classifica male oggetti piccoli.
        m32 = F.interpolate(combined_weight, size=f_sr['stride32'].shape[2:], mode='bilinear')
        cos_sim32 = F.cosine_similarity(f_sr['stride32'], f_hr['stride32'], dim=1, eps=1e-8)
        loss_s32 = (1.0 - cos_sim32).unsqueeze(1)
        total_loss += torch.mean(loss_s32 * (1.0 + m32)) * 0.05

        return total_loss * self.loss_weight

    @staticmethod
    def _instance_norm(x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        std  = x.std(dim=[2, 3], keepdim=True) + 1e-8
        return (x - mean) / std
        
    def _prepare(self, x):
        return x.unsqueeze(0) if x.dim() == 3 else x

    @staticmethod
    def load_clean_state_dict(model, model_path, device):
        print(f"[DDSRNFeatureLoss] Loading Expert Weights: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle dict vs state_dict
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("params", checkpoint))
        else:
            state_dict = checkpoint.state_dict()
        
        # Clean keys (handling compilation prefixes or wrapper prefixes)
        clean_state = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "").replace("module.", "").replace("scorer.", "")
            clean_state[new_key] = v
            
        model.load_state_dict(clean_state, strict=False)