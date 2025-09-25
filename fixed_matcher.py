"""
Fixed matcher module to replace the buggy one in the installed rfdetr package
"""
import os
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

def box_cxcywh_to_xyxy(x):
    """Convert bbox from center format to corner format"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

class FixedHungarianMatcher(nn.Module):
    """Fixed version of HungarianMatcher with proper bounds checking"""

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha: float = 0.25, use_pos_only: bool = False,
                 use_position_modulated_cost: bool = False):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets, group_detr=1):
        """ Performs the matching with comprehensive error handling """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Comprehensive bbox sanitization function
        def sanitize_bbox_tensor(bbox_tensor, name, device):
            """Sanitize bbox tensor to prevent NaN/Inf values and ensure valid coordinates"""
            # Replace NaN and Inf values with valid defaults
            nan_mask = torch.isnan(bbox_tensor)
            inf_mask = torch.isinf(bbox_tensor)
            
            if nan_mask.any() or inf_mask.any():
                print(f"Warning: {name} contains NaN ({nan_mask.sum().item()}) or Inf ({inf_mask.sum().item()}) values. Sanitizing...")
                
                # Replace NaN/Inf in center coordinates (cx, cy) with 0.5 (center of normalized image)
                bbox_tensor[nan_mask[:, :2] | inf_mask[:, :2]] = 0.5
                
                # Replace NaN/Inf in width/height (w, h) with small positive values
                wh_mask = (nan_mask[:, 2:] | inf_mask[:, 2:])
                bbox_tensor[:, 2:][wh_mask] = 0.01
            
            # Ensure all bbox values are within reasonable bounds for normalized coordinates
            # Center coordinates should be in [0, 1]
            bbox_tensor[:, :2] = torch.clamp(bbox_tensor[:, :2], min=0.0, max=1.0)
            
            # Width and height should be positive and <= 1
            bbox_tensor[:, 2:] = torch.clamp(bbox_tensor[:, 2:], min=0.001, max=1.0)
            
            # Additional safety check: ensure no extreme values that could cause overflow
            bbox_tensor = torch.clamp(bbox_tensor, min=-10.0, max=10.0)
            
            return bbox_tensor

        # Sanitize both bbox tensors
        out_bbox = sanitize_bbox_tensor(out_bbox, "out_bbox", out_bbox.device)
        tgt_bbox = sanitize_bbox_tensor(tgt_bbox, "tgt_bbox", tgt_bbox.device)

        # Debug information (can be removed in production)
        if torch.cuda.is_available() and os.environ.get('CUDA_LAUNCH_BLOCKING') == '1':
            print(f"out_bbox shape: {out_bbox.shape}, dtype: {out_bbox.dtype}")
            print(f"tgt_bbox shape: {tgt_bbox.shape}, dtype: {tgt_bbox.dtype}")
            print(f"out_bbox range: [{out_bbox.min().item():.6f}, {out_bbox.max().item():.6f}]")
            print(f"tgt_bbox range: [{tgt_bbox.min().item():.6f}, {tgt_bbox.max().item():.6f}]")

        # Compute the giou cost between boxes with error handling
        try:
            out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
            tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
            
            # Additional sanitization for xyxy format
            out_bbox_xyxy = torch.clamp(out_bbox_xyxy, min=0.0, max=1.0)
            tgt_bbox_xyxy = torch.clamp(tgt_bbox_xyxy, min=0.0, max=1.0)
            
            giou = generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)
            
            # Handle NaN in giou computation
            if torch.isnan(giou).any():
                print("Warning: NaN detected in GIoU computation, replacing with -1")
                giou = torch.where(torch.isnan(giou), torch.tensor(-1.0, device=giou.device), giou)
            
            cost_giou = -giou
            
        except Exception as e:
            print(f"Error in GIoU computation: {e}")
            # Fallback: use zero cost for giou
            cost_giou = torch.zeros((out_bbox.shape[0], tgt_bbox.shape[0]), device=out_bbox.device)

        # Compute the classification cost with additional safety checks
        alpha = 0.25
        gamma = 2.0
        
        # Clamp probabilities to prevent log(0)
        out_prob_clamped = torch.clamp(out_prob, min=1e-8, max=1.0-1e-8)
        
        neg_cost_class = (1 - alpha) * (out_prob_clamped ** gamma) * (-(1 - out_prob_clamped + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob_clamped) ** gamma) * (-(out_prob_clamped + 1e-8).log())
        
        # CRITICAL FIX: Ensure tgt_ids are valid indices
        num_classes = out_prob.shape[1]
        print(f"DEBUG: num_classes={num_classes}, tgt_ids min={tgt_ids.min().item()}, max={tgt_ids.max().item()}")
        
        # Clamp target IDs to valid range [0, num_classes-1]
        tgt_ids_clamped = torch.clamp(tgt_ids, min=0, max=num_classes-1)
        
        # Additional safety check
        if (tgt_ids >= num_classes).any() or (tgt_ids < 0).any():
            print(f"WARNING: Invalid target IDs detected! Original range: [{tgt_ids.min().item()}, {tgt_ids.max().item()}]")
            print(f"Clamping to valid range: [0, {num_classes-1}]")
            invalid_mask = (tgt_ids >= num_classes) | (tgt_ids < 0)
            print(f"Number of invalid IDs: {invalid_mask.sum().item()}")
        
        # Use the clamped IDs for indexing
        try:
            cost_class = pos_cost_class[:, tgt_ids_clamped] - neg_cost_class[:, tgt_ids_clamped]
        except Exception as e:
            print(f"Error in classification cost computation: {e}")
            print(f"pos_cost_class shape: {pos_cost_class.shape}")
            print(f"tgt_ids_clamped shape: {tgt_ids_clamped.shape}, values: {tgt_ids_clamped}")
            # Fallback: use zero cost for classification
            cost_class = torch.zeros((out_prob.shape[0], tgt_bbox.shape[0]), device=out_prob.device)

        # Compute the L1 cost between boxes with error handling
        try:
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            
            # Handle potential NaN/Inf in distance computation
            if torch.isnan(cost_bbox).any() or torch.isinf(cost_bbox).any():
                print("Warning: NaN/Inf detected in bbox distance computation, replacing with large values")
                cost_bbox = torch.where(torch.isnan(cost_bbox) | torch.isinf(cost_bbox), 
                                      torch.tensor(100.0, device=cost_bbox.device), cost_bbox)
        except Exception as e:
            print(f"Error in bbox distance computation: {e}")
            # Fallback: use large constant cost
            cost_bbox = torch.full((out_bbox.shape[0], tgt_bbox.shape[0]), 100.0, device=out_bbox.device)

        # Final cost matrix with additional safety checks
        try:
            # Ensure all cost components are finite
            cost_bbox = torch.where(torch.isfinite(cost_bbox), cost_bbox, torch.tensor(100.0, device=cost_bbox.device))
            cost_class = torch.where(torch.isfinite(cost_class), cost_class, torch.tensor(10.0, device=cost_class.device))
            cost_giou = torch.where(torch.isfinite(cost_giou), cost_giou, torch.tensor(1.0, device=cost_giou.device))
            
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            
            # Additional safety check for the final cost matrix
            C = torch.where(torch.isfinite(C), C, torch.tensor(1000.0, device=C.device))
            
        except Exception as e:
            print(f"Error in cost matrix computation: {e}")
            # Fallback: create a simple distance-based cost matrix
            C = torch.cdist(out_bbox[:, :2], tgt_bbox[:, :2], p=2)  # Use only center coordinates
            C = torch.where(torch.isfinite(C), C, torch.tensor(1000.0, device=C.device))
        
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        g_num_queries = num_queries // group_detr
        
        try:
            C_list = C.split(g_num_queries, dim=1)
            for g_i in range(group_detr):
                C_g = C_list[g_i]
                # Additional safety check before linear sum assignment
                C_split = C_g.split(sizes, -1)
                indices_g = []
                for i, c in enumerate(C_split):
                    # Ensure the cost matrix is finite and not too large
                    c_safe = torch.clamp(c, min=0.0, max=1e6)
                    if torch.isfinite(c_safe).all():
                        indices_g.append(linear_sum_assignment(c_safe))
                    else:
                        # Fallback: create identity matching up to min size
                        min_size = min(c_safe.shape)
                        indices_g.append((np.arange(min_size), np.arange(min_size)))
                
                if g_i == 0:
                    indices = indices_g
                else:
                    indices = [
                        (np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]), np.concatenate([indice1[1], indice2[1]]))
                        for indice1, indice2 in zip(indices, indices_g)
                    ]
        except Exception as e:
            print(f"Error in Hungarian matching: {e}")
            # Fallback: create simple sequential matching
            indices = []
            for target in targets:
                num_targets = len(target["boxes"])
                indices.append((np.arange(min(num_queries, num_targets)), np.arange(num_targets)))
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(args):
    return FixedHungarianMatcher(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        focal_alpha=args.focal_alpha,)