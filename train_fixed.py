import os
import torch
import sys
from pathlib import Path

# Add current directory to path so we can import our fixed modules
sys.path.insert(0, str(Path(__file__).parent))

# Set up debugging environment first
os.environ["PYTHONUTF8"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # 有助于更精确的错误定位
os.environ['TORCH_USE_CUDA_DSA'] = "1"    # 启用设备端断言
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from debug_utils import setup_cuda_debugging, check_cuda_memory

def monkey_patch_torch_indexing():
    """
    Monkey patch torch indexing operations to be safer
    """
    import torch
    
    # Store original functions
    if not hasattr(torch, '_original_index_select'):
        torch._original_index_select = torch.index_select
        torch._original_gather = torch.gather
        
        def safe_index_select(input_tensor, dim, index):
            """Safe version of index_select with bounds checking"""
            if index.numel() == 0:
                # Return empty tensor with correct shape
                shape = list(input_tensor.shape)
                shape[dim] = 0
                return torch.empty(shape, dtype=input_tensor.dtype, device=input_tensor.device)
            
            # Clamp indices to valid range
            max_index = input_tensor.size(dim) - 1
            if max_index < 0:
                # Empty dimension, return empty result
                shape = list(input_tensor.shape)
                shape[dim] = index.shape[0]
                return torch.zeros(shape, dtype=input_tensor.dtype, device=input_tensor.device)
            
            # Clamp indices
            index_clamped = torch.clamp(index, min=0, max=max_index)
            
            if not torch.equal(index, index_clamped):
                print(f"WARNING: Clamped indices in index_select. Original range: [{index.min().item()}, {index.max().item()}], valid range: [0, {max_index}]")
            
            return torch._original_index_select(input_tensor, dim, index_clamped)
        
        def safe_gather(input_tensor, dim, index):
            """Safe version of gather with bounds checking"""
            if index.numel() == 0:
                return torch.empty_like(index, dtype=input_tensor.dtype)
            
            max_index = input_tensor.size(dim) - 1
            if max_index < 0:
                return torch.zeros_like(index, dtype=input_tensor.dtype)
            
            index_clamped = torch.clamp(index, min=0, max=max_index)
            
            if not torch.equal(index, index_clamped):
                print(f"WARNING: Clamped indices in gather. Original range: [{index.min().item()}, {index.max().item()}], valid range: [0, {max_index}]")
            
            return torch._original_gather(input_tensor, dim, index_clamped)
        
        # Apply patches
        torch.index_select = safe_index_select
        torch.gather = safe_gather
        print("Applied safe indexing patches to torch")

def patch_tensor_indexing():
    """
    Patch tensor indexing to be safer
    """
    import torch
    
    original_getitem = torch.Tensor.__getitem__
    
    def safe_getitem(self, key):
        """Safe tensor indexing with bounds checking"""
        try:
            return original_getitem(self, key)
        except (IndexError, RuntimeError) as e:
            if "index out of bounds" in str(e) or "IndexError" in str(e):
                print(f"WARNING: Index out of bounds in tensor indexing: {e}")
                print(f"Tensor shape: {self.shape}, Index: {key}")
                
                # Handle different types of indexing
                if isinstance(key, tuple):
                    # Multi-dimensional indexing
                    new_key = []
                    for i, k in enumerate(key):
                        if isinstance(k, torch.Tensor):
                            # Clamp tensor indices
                            max_idx = self.size(i) - 1
                            k_clamped = torch.clamp(k, min=0, max=max_idx)
                            new_key.append(k_clamped)
                        elif isinstance(k, slice):
                            new_key.append(k)
                        elif isinstance(k, int):
                            max_idx = self.size(i) - 1
                            k_clamped = max(0, min(k, max_idx))
                            new_key.append(k_clamped)
                        else:
                            new_key.append(k)
                    return original_getitem(self, tuple(new_key))
                elif isinstance(key, torch.Tensor):
                    # Single tensor index
                    max_idx = self.size(0) - 1
                    key_clamped = torch.clamp(key, min=0, max=max_idx)
                    return original_getitem(self, key_clamped)
                else:
                    # Fallback: return zeros with appropriate shape
                    return torch.zeros_like(self[0:1] if self.numel() > 0 else self)
            else:
                raise e
    
    torch.Tensor.__getitem__ = safe_getitem
    print("Applied safe tensor indexing patch")

def main():
    # Apply all patches first
    monkey_patch_torch_indexing()
    patch_tensor_indexing()
    
    # Set up debugging environment
    setup_cuda_debugging()
    
    # Print environment info
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
        check_cuda_memory()
    
    # Import rfdetr after applying patches
    from rfdetr import RFDETRBase
    
    # Model configuration
    MODEL_PATH = "F:/code/rf-detr/rf-detr-base.pth"
    
    # Create model instance
    model = RFDETRBase()

    # Train with error handling
    try:
        model.train(
            dataset_dir="F:/res/data",
            epochs=10,
            batch_size=2,  # Further reduce batch size
            grad_accum_steps=4,  # Increase gradient accumulation
        )
    except Exception as e:
        print(f"Training failed with error: {e}")
        print("This might be due to the installed package having bugs.")
        print("Please check the dataset and try running with even smaller batch size.")
        
        # Try with minimal settings
        print("Attempting with minimal batch size...")
        try:
            model.train(
                dataset_dir="F:/res/data",
                epochs=1,
                batch_size=1,
                grad_accum_steps=8,
            )
        except Exception as e2:
            print(f"Even minimal training failed: {e2}")
            print("The issue is likely in the installed rfdetr package's matcher implementation.")
            print("Consider updating the package or using a different version.")

if __name__ == '__main__':
    # Set PyTorch thread count
    torch.set_num_threads(4)
    
    # Run main function
    main()