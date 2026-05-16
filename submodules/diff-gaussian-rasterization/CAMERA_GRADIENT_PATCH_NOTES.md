# Camera-gradient rasterizer patch v1

This patch modifies the `diff-gaussian-rasterization` CUDA extension to compute and return experimental camera-related gradients from the backward pass:

- `dL_dviewmatrix`: partial gradient w.r.t. the 4x4 view matrix
- `dL_dprojmatrix`: gradient w.r.t. the 4x4 full projection matrix from the 2D mean projection path
- `dL_dcampos`: gradient w.r.t. camera center from the SH view-direction path

The current Python wrapper unpacks these tensors but does not yet expose them to autograd, because the original `GaussianRasterizationSettings` passes camera matrices through a `NamedTuple`, not as differentiable inputs to the custom autograd function. A later Python-side patch should pass `viewmatrix`, `projmatrix`, and `campos` explicitly into `_RasterizeGaussians.apply(...)` and return these gradients in `backward()`.

Important limitations of this first CUDA patch:

- The view-matrix gradient is partial: it includes the covariance/depth path through transformed Gaussian means, but not every direct covariance orientation term through `W`.
- The projection-matrix gradient is from the screen-space 2D Gaussian mean path.
- This patch is intended to establish a compileable camera-gradient pathway before wiring learnable SE(3) camera deltas in Python.

Build from inside this folder with your 3DGS conda environment active:

```bash
pip install . --no-build-isolation --force-reinstall
```

or, if your repo normally installs editable submodules:

```bash
pip install -e . --no-build-isolation
```

A quick smoke test after install:

```bash
python - <<'PY'
import torch
from diff_gaussian_rasterization import _C
print('loaded', _C)
PY
```
