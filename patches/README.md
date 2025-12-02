# Patches Directory

This directory contains compatibility patches for Wan2.2 deployment on cloud platforms.

## apply_patches.py

FlashAttention compatibility patches that must be applied **BEFORE** importing any Wan2.2 modules.

### What it does:
1. Disables FlashAttention flags in `wan/modules/attention.py`
2. Redirects flash_attention imports to standard attention() in `wan/s2v/model_s2v.py`
3. Applies same aliasing to shared `wan/modules/model.py`

### Usage:
```python
from patches.apply_patches import apply_flashattention_patches
apply_flashattention_patches()

# Now safe to import Wan2.2
from wan.s2v.model_s2v import ...
```

### Why this is needed:
FlashAttention requires complex CUDA compilation that fails on most cloud platforms. This patch allows Wan2.2 to run with standard PyTorch attention mechanisms instead.
