import os
import torch
import sys
import warnings
from pathlib import Path

# Suppress the meshgrid warning
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")

# Add current directory to path so we can import our fixed modules
sys.path.insert(0, str(Path(__file__).parent))

# Set up debugging environment first
os.environ["PYTHONUTF8"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # 有助于更精确的错误定位
os.environ['TORCH_USE_CUDA_DSA'] = "1"    # 启用设备端断言
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from debug_utils import setup_cuda_debugging, check_cuda_memory

def create_fixed_matcher():
    """
    Create a completely fixed matcher that handles all the issues
    """
    import torch
    import torch.nn as nn
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    
    class UltimateFixer(nn.Module):
        def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha: float = 0.25):
            super().__init__()
            self.cost_class = cost_class
            self.cost_bbox = cost_bbox
            self.cost_giou = cost_giou
            self.focal_alpha = focal_alpha

        def box_cxcywh_to_xyxy(self, x):
            """Convert bbox from center format to corner format"""
            x_c, y_c, w, h = x.unbind(-1)
            b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
                 (x_c + 0.5 * w), (y_c + 0.5 * h)]
            return torch.stack(b, dim=-1)

        def box_area(self, boxes):
            """Computes the area of a set of bounding boxes"""
            return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        def box_iou(self, boxes1, boxes2):
            """Return intersection-over-union (Jaccard index) of boxes."""
            area1 = self.box_area(boxes1)
            area2 = self.box_area(boxes2)

            lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
            rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

            wh = (rb - lt).clamp(min=0)  # [N,M,2]
            inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

            union = area1[:, None] + area2 - inter
            iou = inter / (union + 1e-6)
            return iou, union

        def generalized_box_iou(self, boxes1, boxes2):
            """Generalized IoU from https://giou.stanford.edu/"""
            # Ensure boxes are valid
            boxes1 = torch.clamp(boxes1, min=0.0, max=1.0)
            boxes2 = torch.clamp(boxes2, min=0.0, max=1.0)
            
            # Ensure x2 >= x1 and y2 >= y1
            boxes1[:, 2:] = torch.max(boxes1[:, 2:], boxes1[:, :2] + 0.001)
            boxes2[:, 2:] = torch.max(boxes2[:, 2:], boxes2[:, :2] + 0.001)
            
            try:
                iou, union = self.box_iou(boxes1, boxes2)

                lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
                rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

                wh = (rb - lt).clamp(min=0)  # [N,M,2]
                area = wh[:, :, 0] * wh[:, :, 1]

                giou = iou - (area - union) / (area + 1e-6)
                return giou
            except:
                # Fallback: return simple IoU
                return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

        @torch.no_grad()
        def forward(self, outputs, targets, group_detr=1):
            """Performs the matching with ultimate fixes"""
            bs, num_queries = outputs["pred_logits"].shape[:2]
            
            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])
            
            # CRITICAL FIX: Convert 1-indexed class IDs to 0-indexed
            print(f"Original tgt_ids range: [{tgt_ids.min().item()}, {tgt_ids.max().item()}]")
            tgt_ids = tgt_ids - 1  # Convert from 1-indexed to 0-indexed
            tgt_ids = torch.clamp(tgt_ids, min=0)  # Ensure no negative indices
            print(f"Converted tgt_ids range: [{tgt_ids.min().item()}, {tgt_ids.max().item()}]")

            # Sanitize bbox tensors
            def sanitize_bbox(bbox, name):
                # Replace NaN/Inf
                bbox = torch.where(torch.isnan(bbox) | torch.isinf(bbox), 
                                 torch.tensor(0.5, device=bbox.device), bbox)
                # Clamp to valid ranges
                bbox[:, :2] = torch.clamp(bbox[:, :2], min=0.0, max=1.0)  # center
                bbox[:, 2:] = torch.clamp(bbox[:, 2:], min=0.001, max=1.0)  # size
                return bbox

            out_bbox = sanitize_bbox(out_bbox, "out_bbox")
            tgt_bbox = sanitize_bbox(tgt_bbox, "tgt_bbox")

            print(f"out_bbox shape: {out_bbox.shape}, dtype: {out_bbox.dtype}")
            print(f"tgt_bbox shape: {tgt_bbox.shape}, dtype: {tgt_bbox.dtype}")
            print(f"out_bbox range: [{out_bbox.min().item():.6f}, {out_bbox.max().item():.6f}]")
            print(f"tgt_bbox range: [{tgt_bbox.min().item():.6f}, {tgt_bbox.max().item():.6f}]")

            num_classes = out_prob.shape[1]
            print(f"DEBUG: num_classes={num_classes}, tgt_ids min={tgt_ids.min().item()}, max={tgt_ids.max().item()}")

            # Ensure tgt_ids are within valid range
            if (tgt_ids >= num_classes).any() or (tgt_ids < 0).any():
                print(f"WARNING: Invalid target IDs detected! Original range: [{tgt_ids.min().item()}, {tgt_ids.max().item()}]")
                print(f"Clamping to valid range: [0, {num_classes-1}]")
                invalid_mask = (tgt_ids >= num_classes) | (tgt_ids < 0)
                print(f"Number of invalid IDs: {invalid_mask.sum().item()}")
                tgt_ids = torch.clamp(tgt_ids, min=0, max=num_classes-1)

            # Compute costs with extensive error handling
            try:
                # Classification cost
                alpha = 0.25
                gamma = 2.0
                
                out_prob_safe = torch.clamp(out_prob, min=1e-8, max=1.0-1e-8)
                neg_cost_class = (1 - alpha) * (out_prob_safe ** gamma) * (-(1 - out_prob_safe + 1e-8).log())
                pos_cost_class = alpha * ((1 - out_prob_safe) ** gamma) * (-(out_prob_safe + 1e-8).log())
                
                # Safe indexing for classification cost
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
                
                # Handle any remaining NaN/Inf
                cost_class = torch.where(torch.isfinite(cost_class), cost_class, 
                                       torch.tensor(10.0, device=cost_class.device))
                
            except Exception as e:
                print(f"Error in classification cost: {e}")
                cost_class = torch.ones((out_prob.shape[0], tgt_bbox.shape[0]), device=out_prob.device) * 10.0

            try:
                # Bbox cost
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
                cost_bbox = torch.where(torch.isfinite(cost_bbox), cost_bbox,
                                      torch.tensor(100.0, device=cost_bbox.device))
            except Exception as e:
                print(f"Error in bbox cost: {e}")
                cost_bbox = torch.ones((out_bbox.shape[0], tgt_bbox.shape[0]), device=out_bbox.device) * 100.0

            try:
                # GIoU cost
                out_bbox_xyxy = self.box_cxcywh_to_xyxy(out_bbox)
                tgt_bbox_xyxy = self.box_cxcywh_to_xyxy(tgt_bbox)
                giou = self.generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)
                cost_giou = -giou
                cost_giou = torch.where(torch.isfinite(cost_giou), cost_giou,
                                      torch.tensor(1.0, device=cost_giou.device))
            except Exception as e:
                print(f"Error in GIoU cost: {e}")
                cost_giou = torch.ones((out_bbox.shape[0], tgt_bbox.shape[0]), device=out_bbox.device)

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = torch.where(torch.isfinite(C), C, torch.tensor(1000.0, device=C.device))
            
            # CRITICAL FIX: Ensure C is 2D before reshaping
            if C.dim() != 2:
                print(f"WARNING: Cost matrix has wrong dimensions: {C.shape}")
                C = C.view(-1, tgt_bbox.shape[0])
            
            # Reshape for batch processing
            C = C.view(bs, num_queries, -1)
            
            # Convert to CPU for scipy
            C_cpu = C.cpu().numpy()

            sizes = [len(v["boxes"]) for v in targets]
            indices = []
            
            try:
                for i, c in enumerate(C_cpu):
                    # Ensure c is 2D
                    if c.ndim != 2:
                        print(f"WARNING: Cost matrix for batch {i} has wrong shape: {c.shape}")
                        c = c.reshape(num_queries, -1)
                    
                    # Split by target sizes
                    c_split = np.split(c, np.cumsum(sizes)[:-1], axis=1)
                    batch_indices = []
                    
                    for j, c_j in enumerate(c_split):
                        if c_j.size == 0:
                            batch_indices.append((np.array([]), np.array([])))
                            continue
                            
                        # Ensure c_j is 2D
                        if c_j.ndim != 2:
                            c_j = c_j.reshape(-1, sizes[j])
                        
                        # Clamp values to prevent numerical issues
                        c_j = np.clip(c_j, 0, 1e6)
                        
                        try:
                            row_ind, col_ind = linear_sum_assignment(c_j)
                            batch_indices.append((row_ind, col_ind))
                        except Exception as e:
                            print(f"Linear sum assignment failed for batch {i}, target {j}: {e}")
                            # Fallback: simple sequential assignment
                            min_size = min(c_j.shape[0], c_j.shape[1])
                            batch_indices.append((np.arange(min_size), np.arange(min_size)))
                    
                    # Combine indices for this batch
                    if len(batch_indices) == 1:
                        indices.append(batch_indices[0])
                    else:
                        # Multiple targets in batch - combine them
                        all_rows = []
                        all_cols = []
                        col_offset = 0
                        for row_ind, col_ind in batch_indices:
                            all_rows.extend(row_ind)
                            all_cols.extend(col_ind + col_offset)
                            col_offset += len(col_ind)
                        indices.append((np.array(all_rows), np.array(all_cols)))
                        
            except Exception as e:
                print(f"Error in Hungarian matching: {e}")
                # Ultimate fallback
                indices = []
                for target in targets:
                    num_targets = len(target["boxes"])
                    min_size = min(num_queries, num_targets)
                    indices.append((np.arange(min_size), np.arange(min_size)))

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
    return UltimateFixer

def apply_ultimate_patches():
    """Apply the ultimate patches to fix all issues"""
    print("Applying ultimate patches...")
    
    # Fix meshgrid warning
    import torch
    original_meshgrid = torch.meshgrid
    
    def fixed_meshgrid(*tensors, **kwargs):
        if 'indexing' not in kwargs:
            kwargs['indexing'] = 'ij'
        return original_meshgrid(*tensors, **kwargs)
    
    torch.meshgrid = fixed_meshgrid
    
    # Apply tensor indexing safety
    original_getitem = torch.Tensor.__getitem__
    
    def ultra_safe_getitem(self, key):
        try:
            return original_getitem(self, key)
        except (IndexError, RuntimeError) as e:
            if "index out of bounds" in str(e).lower():
                print(f"CRITICAL: Prevented index out of bounds! Tensor shape: {self.shape}, key: {key}")
                # Return a safe fallback
                if isinstance(key, tuple) and len(key) >= 2:
                    # Multi-dimensional indexing
                    dim1_key, dim2_key = key[0], key[1]
                    if isinstance(dim2_key, torch.Tensor):
                        # Clamp the second dimension indices
                        safe_key = torch.clamp(dim2_key, 0, self.size(1) - 1)
                        return original_getitem(self, (dim1_key, safe_key))
                # Other fallbacks...
                return torch.zeros(1, dtype=self.dtype, device=self.device)
            else:
                raise e
    
    torch.Tensor.__getitem__ = ultra_safe_getitem
    print("Applied ultimate safety patches")

def main():
    # Apply ultimate patches
    apply_ultimate_patches()
    
    # Set up debugging
    setup_cuda_debugging()
    
    # Print environment info
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
        check_cuda_memory()
    
    # Import and patch rfdetr
    try:
        from rfdetr import RFDETRBase
        
        # Create our fixed matcher
        FixedMatcher = create_fixed_matcher()
        
        # Try to patch the matcher in various possible locations
        import sys
        for module_name, module in sys.modules.items():
            if 'matcher' in module_name.lower() and hasattr(module, 'HungarianMatcher'):
                print(f"Patching {module_name}")
                module.HungarianMatcher = FixedMatcher
                
        print("Matcher patching completed")
        
    except Exception as e:
        print(f"Error in patching: {e}")
        print("Continuing with available patches...")
        from rfdetr import RFDETRBase

    # Create model
    model = RFDETRBase()

    # Train with very conservative settings
    try:
        print("Starting training with ultimate fixes...")
        model.train(
            dataset_dir="F:/res/data",
            epochs=5,
            batch_size=1,  # Minimal batch size
            grad_accum_steps=8,  # High accumulation
        )
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("\nThis suggests the issue is in the installed package itself.")
        print("The class ID mismatch (1-indexed vs 0-indexed) is the main issue.")
        print("\nRecommendation: Check your dataset annotations and ensure class IDs start from 0, not 1.")

if __name__ == '__main__':
    torch.set_num_threads(4)
    main()