#!/usr/bin/env python3
"""
Debug utilities for RF-DETR training
"""
import torch
import os

def setup_cuda_debugging():
    """Set up environment variables for CUDA debugging"""
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    print("CUDA debugging environment set up")

def validate_tensor(tensor, name="tensor", check_finite=True, check_range=None):
    """
    Validate tensor for common issues
    
    Args:
        tensor: PyTorch tensor to validate
        name: Name for logging
        check_finite: Check for NaN/Inf values
        check_range: Tuple of (min, max) expected values
    """
    if tensor is None:
        print(f"Warning: {name} is None")
        return False
    
    if not isinstance(tensor, torch.Tensor):
        print(f"Warning: {name} is not a tensor, type: {type(tensor)}")
        return False
    
    print(f"{name} shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
    
    if check_finite:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        
        if nan_count > 0:
            print(f"Warning: {name} contains {nan_count} NaN values")
            return False
        
        if inf_count > 0:
            print(f"Warning: {name} contains {inf_count} Inf values")
            return False
    
    if check_range:
        min_val, max_val = check_range
        tensor_min = tensor.min().item()
        tensor_max = tensor.max().item()
        
        print(f"{name} range: [{tensor_min:.6f}, {tensor_max:.6f}]")
        
        if tensor_min < min_val or tensor_max > max_val:
            print(f"Warning: {name} values outside expected range [{min_val}, {max_val}]")
            return False
    
    return True

def sanitize_bbox_tensor(bbox_tensor, name="bbox"):
    """
    Sanitize bbox tensor to ensure valid values
    
    Args:
        bbox_tensor: Tensor of shape [..., 4] in cxcywh format
        name: Name for logging
        
    Returns:
        Sanitized bbox tensor
    """
    if bbox_tensor is None:
        return None
    
    original_shape = bbox_tensor.shape
    device = bbox_tensor.device
    
    # Flatten for easier processing
    bbox_flat = bbox_tensor.view(-1, 4)
    
    # Replace NaN/Inf values
    nan_mask = torch.isnan(bbox_flat)
    inf_mask = torch.isinf(bbox_flat)
    
    if nan_mask.any() or inf_mask.any():
        print(f"Sanitizing {name}: NaN={nan_mask.sum().item()}, Inf={inf_mask.sum().item()}")
        
        # Replace NaN/Inf in center coordinates with 0.5
        bbox_flat[nan_mask[:, :2] | inf_mask[:, :2]] = 0.5
        
        # Replace NaN/Inf in width/height with small positive values
        wh_mask = (nan_mask[:, 2:] | inf_mask[:, 2:])
        bbox_flat[:, 2:][wh_mask] = 0.01
    
    # Clamp to reasonable ranges
    bbox_flat[:, :2] = torch.clamp(bbox_flat[:, :2], min=0.0, max=1.0)  # center coordinates
    bbox_flat[:, 2:] = torch.clamp(bbox_flat[:, 2:], min=0.001, max=1.0)  # width/height
    
    return bbox_flat.view(original_shape)

def check_cuda_memory():
    """Check CUDA memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"CUDA Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")
        
        # Clear cache if memory usage is high
        if allocated > 8.0:  # If using more than 8GB
            print("High memory usage detected, clearing CUDA cache...")
            torch.cuda.empty_cache()

def safe_model_forward(model, inputs, targets=None):
    """
    Safely run model forward pass with error handling
    
    Args:
        model: The model to run
        inputs: Input data
        targets: Target data (optional)
        
    Returns:
        Model outputs or None if error occurred
    """
    try:
        if targets is not None:
            outputs = model(inputs, targets)
        else:
            outputs = model(inputs)
        
        # Validate outputs
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    if not validate_tensor(value, f"output_{key}"):
                        print(f"Invalid tensor in model output: {key}")
                        return None
        
        return outputs
        
    except RuntimeError as e:
        if "CUDA" in str(e) or "index out of bounds" in str(e):
            print(f"CUDA error in model forward: {e}")
            check_cuda_memory()
            return None
        else:
            raise e

if __name__ == "__main__":
    # Test the utilities
    setup_cuda_debugging()
    check_cuda_memory()
    
    # Test with sample tensors
    test_bbox = torch.tensor([[0.5, 0.5, 0.2, 0.2], [float('nan'), 0.3, 0.1, float('inf')]])
    print("Original bbox:", test_bbox)
    
    sanitized = sanitize_bbox_tensor(test_bbox, "test_bbox")
    print("Sanitized bbox:", sanitized)
    
    validate_tensor(sanitized, "sanitized_bbox", check_range=(0.0, 1.0))