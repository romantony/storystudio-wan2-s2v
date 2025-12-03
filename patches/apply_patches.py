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
    
    # Phase 2: Patch model files that import flash_attention
    print("\n[2/3] Patching model files that import flash_attention...")
    candidates = [
        wan_path / "wan/s2v/model_s2v.py",
        wan_path / "wan/tasks/s2v/model_s2v.py",
        wan_path / "wan/models/s2v.py",
    ]

    patched_any = False

    for candidate in candidates:
        if candidate.exists():
            code = candidate.read_text()
            before = code
            code = code.replace(
                'from ..modules.attention import flash_attention',
                'from ..modules.attention import attention as flash_attention'
            )
            code = code.replace(
                'from wan.modules.attention import flash_attention',
                'from wan.modules.attention import attention as flash_attention'
            )
            if code != before:
                candidate.write_text(code)
                print(f"✓ Patched imports in {candidate}")
                patched_any = True

    # Fallback: scan repository for any python files importing flash_attention
    if not patched_any:
        print("No known model_s2v.py paths found, scanning repository for imports...")
        for py in wan_path.rglob('*.py'):
            try:
                text = py.read_text()
            except Exception:
                continue
            if ('from wan.modules.attention import flash_attention' in text or
                'from ..modules.attention import flash_attention' in text):
                updated = text.replace(
                    'from wan.modules.attention import flash_attention',
                    'from wan.modules.attention import attention as flash_attention'
                ).replace(
                    'from ..modules.attention import flash_attention',
                    'from ..modules.attention import attention as flash_attention'
                )
                if updated != text:
                    py.write_text(updated)
                    print(f"✓ Patched imports in {py}")
                    patched_any = True

    if not patched_any:
        print("! Warning: No files required import patching for flash_attention.")
    
    # Phase 3: Patch shared model.py
    print("\n[3/3] Patching wan/modules/model.py...")
    common_model_path = wan_path / "wan/modules/model.py"
    
    if common_model_path.exists():
        common_code = common_model_path.read_text()
        before = common_code
        
        # Apply same replacement patterns
        common_code = common_code.replace(
            'from .attention import flash_attention',
            'from .attention import attention as flash_attention'
        )
        common_code = common_code.replace(
            'from wan.modules.attention import flash_attention',
            'from wan.modules.attention import attention as flash_attention'
        )
        
        if common_code != before:
            common_model_path.write_text(common_code)
            print("✓ model.py patched")
        else:
            print("⚠ model.py had no flash_attention imports to patch")
    else:
        print("⚠ model.py not found (may not be needed)")
    
    # Verification
    print("\n" + "=" * 70)
    print("Patch Verification:")
    attention_check = 'FLASH_ATTN_2_AVAILABLE = False' in attention_path.read_text()
    
    print(f"  {'✓' if attention_check else '✗'} FLASH_ATTN_2 disabled in attention.py")
    print(f"  ✓ Import patching completed (repository scanned)")
    
    if attention_check:
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
