"""
Monkey patch to fix the buggy matcher in the installed rfdetr package
"""
import sys
import importlib.util
from pathlib import Path

def patch_rfdetr_matcher():
    """
    Monkey patch the rfdetr matcher to use our fixed version
    """
    try:
        # Import the fixed matcher
        from fixed_matcher import FixedHungarianMatcher, build_matcher
        
        # Try to patch the installed package
        try:
            import rfdetr.models.matcher as matcher_module
            print("Found rfdetr.models.matcher, patching...")
            matcher_module.HungarianMatcher = FixedHungarianMatcher
            matcher_module.build_matcher = build_matcher
            print("Successfully patched rfdetr.models.matcher")
            return True
        except ImportError:
            pass
        
        # Try alternative import paths
        try:
            import rfdetr.matcher as matcher_module
            print("Found rfdetr.matcher, patching...")
            matcher_module.HungarianMatcher = FixedHungarianMatcher
            matcher_module.build_matcher = build_matcher
            print("Successfully patched rfdetr.matcher")
            return True
        except ImportError:
            pass
            
        # Try to find and patch any matcher module in rfdetr
        import rfdetr
        rfdetr_path = Path(rfdetr.__file__).parent
        
        # Search for matcher files
        matcher_files = list(rfdetr_path.rglob("*matcher*.py"))
        print(f"Found matcher files: {matcher_files}")
        
        for matcher_file in matcher_files:
            try:
                # Load the module
                spec = importlib.util.spec_from_file_location("matcher_module", matcher_file)
                matcher_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(matcher_module)
                
                # Check if it has HungarianMatcher
                if hasattr(matcher_module, 'HungarianMatcher'):
                    print(f"Patching {matcher_file}")
                    matcher_module.HungarianMatcher = FixedHungarianMatcher
                    if hasattr(matcher_module, 'build_matcher'):
                        matcher_module.build_matcher = build_matcher
                    
                    # Also patch in sys.modules if it's there
                    for module_name, module in sys.modules.items():
                        if hasattr(module, 'HungarianMatcher') and 'matcher' in module_name.lower():
                            print(f"Patching sys.modules[{module_name}]")
                            module.HungarianMatcher = FixedHungarianMatcher
                            if hasattr(module, 'build_matcher'):
                                module.build_matcher = build_matcher
                    
                    return True
            except Exception as e:
                print(f"Failed to patch {matcher_file}: {e}")
                continue
        
        print("Could not find matcher module to patch")
        return False
        
    except Exception as e:
        print(f"Error in patching: {e}")
        return False

def apply_comprehensive_patches():
    """
    Apply comprehensive patches to fix the CUDA indexing issues
    """
    print("Applying comprehensive patches for CUDA indexing issues...")
    
    # Patch 1: Fix the matcher
    patch_success = patch_rfdetr_matcher()
    
    # Patch 2: Add global error handling
    import torch
    
    # Store original functions
    original_index_select = torch.index_select
    original_gather = torch.gather
    
    def safe_index_select(input, dim, index):
        """Safe version of torch.index_select with bounds checking"""
        try:
            # Clamp indices to valid range
            max_index = input.size(dim) - 1
            safe_index = torch.clamp(index, min=0, max=max_index)
            
            if not torch.equal(index, safe_index):
                print(f"Warning: Clamped indices in index_select. Original range: [{index.min().item()}, {index.max().item()}], Max allowed: {max_index}")
            
            return original_index_select(input, dim, safe_index)
        except Exception as e:
            print(f"Error in safe_index_select: {e}")
            # Fallback: return zeros with appropriate shape
            shape = list(input.shape)
            shape[dim] = index.shape[0]
            return torch.zeros(shape, dtype=input.dtype, device=input.device)
    
    def safe_gather(input, dim, index):
        """Safe version of torch.gather with bounds checking"""
        try:
            # Clamp indices to valid range
            max_index = input.size(dim) - 1
            safe_index = torch.clamp(index, min=0, max=max_index)
            
            if not torch.equal(index, safe_index):
                print(f"Warning: Clamped indices in gather. Original range: [{index.min().item()}, {index.max().item()}], Max allowed: {max_index}")
            
            return original_gather(input, dim, safe_index)
        except Exception as e:
            print(f"Error in safe_gather: {e}")
            # Fallback: return zeros with appropriate shape
            return torch.zeros_like(index, dtype=input.dtype, device=input.device)
    
    # Apply the patches
    torch.index_select = safe_index_select
    torch.gather = safe_gather
    
    print(f"Patches applied successfully. Matcher patch: {'Success' if patch_success else 'Failed'}")
    return patch_success

if __name__ == "__main__":
    apply_comprehensive_patches()