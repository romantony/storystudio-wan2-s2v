"""
FlashAttention compatibility patches for Wan2.2
Must be applied BEFORE any Wan2.2 imports

This patching strategy forces Wan2.2 to use standard attention instead of FlashAttention,
which fails to build on most cloud platforms including RunPod.
"""
from pathlib import Path
import sys

def apply_flashattention_patches(wan_dir="/workspace/Wan2.2"):
    """
    Apply three-phase patching strategy for FlashAttention compatibility
    
    Phase 1: Disable FLASH_ATTN_2_AVAILABLE flag in attention.py
    Phase 2: Alias flash_attention to attention() in model_s2v.py
    Phase 3: Apply same aliasing to shared model.py
    
    Args:
        wan_dir: Path to Wan2.2 repository
        
    Returns:
        bool: True if all patches applied successfully
    """
    print("=" * 70)
    print("Applying FlashAttention Compatibility Patches")
    print("=" * 70)
    print(f"\nChecking for Wan2.2 at: {wan_dir}")
    
    wan_path = Path(wan_dir)
    if not wan_path.exists():
        # List workspace contents for debugging
        workspace = Path("/workspace")
        if workspace.exists():
            print(f"\nWorkspace contents: {list(workspace.iterdir())}")
        raise RuntimeError(f"Wan2.2 directory not found: {wan_dir}")
    
    print(f"✓ Found Wan2.2 directory")
    print(f"Contents: {list(wan_path.iterdir())[:10]}")
    
    # Phase 1: Patch attention.py
    print("\n[1/3] Patching wan/modules/attention.py...")
    attention_path = wan_path / "wan/modules/attention.py"
    
    if not attention_path.exists():
        raise RuntimeError(f"attention.py not found at {attention_path}")
    
    attention_code = attention_path.read_text()
    
    # Force disable FlashAttention flags
    attention_code = attention_code.replace(
        'FLASH_ATTN_2_AVAILABLE = importlib.util.find_spec("flash_attn") is not None',
        'FLASH_ATTN_2_AVAILABLE = False  # [PATCHED] Forced to False'
    )
    
    # Remove assertion that blocks execution
    if 'assert FLASH_ATTN_2_AVAILABLE' in attention_code:
        lines = attention_code.split('\n')
        attention_code = '\n'.join([
            line if 'assert FLASH_ATTN_2_AVAILABLE' not in line 
            else '    pass  # [PATCHED] Assertion removed'
            for line in lines
        ])
    
    # Add shim for flash_attention import
    if 'def flash_attention(' not in attention_code:
        shim = '''

def flash_attention(*args, **kwargs):
    """[PATCHED] Shim that redirects to attention()"""
    return attention(*args, **kwargs)
'''
        attention_code += shim
    
    attention_path.write_text(attention_code)
    print("✓ attention.py patched")
    
    # Phase 2: Patch model_s2v.py
    print("\n[2/3] Patching wan/s2v/model_s2v.py...")
    model_path = wan_path / "wan/s2v/model_s2v.py"
    
    if not model_path.exists():
        raise RuntimeError(f"model_s2v.py not found at {model_path}")
    
    model_code = model_path.read_text()
    
    # Replace flash_attention imports
    import_patterns = [
        'from ..modules.attention import flash_attention',
        'from wan.modules.attention import flash_attention',
    ]
    for pattern in import_patterns:
        if pattern in model_code:
            model_code = model_code.replace(
                pattern,
                'from ..modules.attention import attention as flash_attention  # [PATCHED]'
            )
    
    model_path.write_text(model_code)
    print("✓ model_s2v.py patched")
    
    # Phase 3: Patch shared model.py
    print("\n[3/3] Patching wan/modules/model.py...")
    common_model_path = wan_path / "wan/modules/model.py"
    
    if common_model_path.exists():
        common_code = common_model_path.read_text()
        
        for pattern in import_patterns:
            if pattern in common_code:
                common_code = common_code.replace(
                    pattern,
                    'from .attention import attention as flash_attention  # [PATCHED]'
                )
        
        common_model_path.write_text(common_code)
        print("✓ model.py patched")
    else:
        print("⚠ model.py not found (may not be needed)")
    
    # Verification
    print("\n" + "=" * 70)
    print("Patch Verification:")
    attention_check = 'FLASH_ATTN_2_AVAILABLE = False' in attention_path.read_text()
    model_check = 'attention as flash_attention' in model_path.read_text()
    
    print(f"  {'✓' if attention_check else '✗'} FLASH_ATTN_2 disabled in attention.py")
    print(f"  {'✓' if model_check else '✗'} model_s2v.py using attention fallback")
    
    if attention_check and model_check:
        print("\n✓ All patches applied successfully!")
        print("=" * 70)
        return True
    else:
        print("\n✗ Some patches failed - check manually")
        print("=" * 70)
        return False

if __name__ == "__main__":
    success = apply_flashattention_patches()
    sys.exit(0 if success else 1)
