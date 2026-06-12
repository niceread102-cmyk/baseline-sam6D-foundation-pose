# 6D Object Pose Estimation with SAM-6D and FoundationPose

This repository wraps [SAM-6D](https://github.com/JiehongLin/SAM-6D) and [FoundationPose](https://github.com/NVlabs/FoundationPose) into two clean Python classes for zero-shot 6D object pose estimation from RGB-D images. No training is required — only a CAD model of the object.

See `sam6d_example.ipynb` for a full end-to-end walkthrough.

## How it works

**SAM-6D** (`src/sam6d.py`) runs a two-stage pipeline:
1. **ISM** — SAM generates mask proposals; DINOv2 scores them against rendered CAD templates via semantic, appearance, and geometric similarity.
2. **PEM** — A ViT-based pose estimation model takes the top proposals and returns 6DoF poses (R, t) in millimetres.

**FoundationPose** (`src/foundationpose.py`) refines the SAM-6D initial poses via iterative RGB-D render-and-compare, then tracks registered objects on subsequent frames without re-running SAM-6D.

---

## System Prerequisites

- **CUDA 11.8+** — tested on CUDA 12.9 (sm_86 / RTX A6000)
- **Python 3.10–3.12** — tested on 3.12
- **cmake ≥ 3.15**, a C++14 compiler, **Eigen3**, **pybind11**, **OpenMP** — needed to build the FoundationPose `mycpp` C++ extension
- **Blender** — BlenderProc wraps it for template rendering; BlenderProc downloads a self-contained Blender binary automatically on first run
- **git** — needed to install `nvdiffrast` from GitHub

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd cnos
```

### 2. Install PyTorch (match CUDA version to your driver)

```bash
# CUDA 12.4 example — check https://pytorch.org/get-started/locally/ for other versions
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install nvdiffrast (requires `nvcc` on PATH)

```bash
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
```

`nvcc` comes with the CUDA Toolkit (not just the driver). If it is not on PATH, set `CUDA_HOME` to your toolkit path (e.g. `/usr/local/cuda`) before running the above.

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Build the FoundationPose C++ extension

FoundationPose uses a small C++ library (`mycpp`) for pose hypothesis clustering. Build it once:

```bash
cd FoundationPose-main/mycpp
mkdir -p build && cd build
cmake .. -DPYTHON_EXECUTABLE=$(which python)
make -j$(nproc)
cd ../../..
```

If you are using a conda environment, you can alternatively run:

```bash
bash FoundationPose-main/build_all_conda.sh
```

### 6. Verify the build

```bash
python -c "
import sys
sys.path.insert(0, 'FoundationPose-main')
import glob, pathlib
so = list(pathlib.Path('FoundationPose-main/mycpp/build').glob('mycpp*.so'))
print('mycpp OK:', so[0].name if so else 'NOT FOUND')
"
```

---

## Running the Notebook

### Generic RGB-D input

```bash
jupyter notebook sam6d_example.ipynb
```

In the **Configuration** cell, update these two variables:

| Variable | Description |
|----------|-------------|
| `CAD_PATH` | Path to your object's CAD file (`.ply`, `.stl`, `.obj`, …) |
| `DEPTH_SCALE` | Multiply raw depth pixels → mm. Use `1.0` if depth is already in mm, `1000.0` for uint16 RealSense depth |

Everything else (model weight downloads, template rendering, descriptor caching) happens automatically on first run.

### Photoneo PhoXi 3D scanner

A dedicated notebook is provided for Photoneo PhoXi scanners:

```bash
jupyter notebook sam6d_example_photoneo.ipynb
```

This notebook reads directly from a `.praw` file via a virtual file camera, or falls back to exported TIF/PNG files if PhoXi Control is not available.

**Additional requirement — `phoxi_api`:**

`phoxi_api` is already listed in `requirements.txt` and installed with the rest of the dependencies:

```bash
pip install phoxi_api
```

**PhoXi Control must be running** when using the live path. Start it before launching the notebook:

```bash
/usr/bin/PhoXiControl &
```

If PhoXi Control is not running (or `phoxi_api` is unavailable), the notebook falls back to loading depth and texture directly from exported TIF/PNG files — set the `PREFIX` variable and fill in `FX`, `FY`, `CX`, `CY` from the calibration panel in PhoXi Control (Camera → Calibration → Camera Matrix).

---

## Model Weights

All weights are auto-downloaded on the first run. Locations after download:

| Weight | Size | Path |
|--------|------|------|
| SAM ViT-H | ~2.5 GB | `SAM-6D-main/SAM-6D/Instance_Segmentation_Model/checkpoints/segment-anything/` |
| DINOv2 ViT-L/14 | ~1.2 GB | `SAM-6D-main/SAM-6D/Instance_Segmentation_Model/checkpoints/dinov2/` |
| SAM-6D PEM | ~300 MB | `SAM-6D-main/SAM-6D/Pose_Estimation_Model/checkpoints/` |
| MAE ViT-Base backbone | ~330 MB | `checkpoints/mae_pretrain_vit_base.pth` |
| FoundationPose refiner | ~150 MB | `FoundationPose-main/weights/2023-10-28-18-33-37/` |
| FoundationPose scorer | ~150 MB | `FoundationPose-main/weights/2024-01-11-20-02-45/` |

SAM, DINOv2, and MAE download via `wget`. PEM and FoundationPose weights download via `gdown` from Google Drive.

**If gdown fails (Google Drive quota exceeded)**, download manually:

- **SAM-6D PEM**: `gdown 1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7 -O <pem-path>`
- **FoundationPose weights**: https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i  
  Place the `2023-10-28-18-33-37` and `2024-01-11-20-02-45` folders under `FoundationPose-main/weights/`.

---

## Template Rendering

On first run, `SAM6D` renders 42 BlenderProc views of your CAD model into:

```
sam6d_templates/<mesh_stem>/templates/
```

This takes ~1–2 minutes and requires BlenderProc (installed via `requirements.txt`). Subsequent runs skip rendering because the outputs are cached. Pass `reset_templates=True` to `SAM6D(...)` to force a re-render.

---

## Repository Structure

```
sam6d_example.ipynb              # End-to-end usage notebook (generic RGB-D)
sam6d_example_photoneo.ipynb     # Photoneo PhoXi scanner notebook
requirements.txt                 # Python dependencies
src/
  sam6d.py                       # SAM6D class — ISM + PEM wrapper
  foundationpose.py              # FoundationPoseRefiner class
SAM-6D-main/                     # SAM-6D source (do not modify)
FoundationPose-main/             # FoundationPose source (do not modify)
checkpoints/                     # MAE ViT-Base backbone checkpoint (auto-downloaded)
```

---

## Troubleshooting

**`mycpp C++ extension not found`**  
Run the cmake/make build step (Section 5 above). If Python version changes, rebuild.

**`BlenderProc rendering failed`**  
Run the render command manually to see Blender's error output:
```bash
blenderproc run SAM-6D-main/SAM-6D/Render/render_custom_templates.py \
  --output_dir ./sam6d_templates --cad_path <your-cad-file>
```

**`gdown` quota / download failure**  
Download weights manually using the links in the Model Weights section above.

**CUDA out of memory**  
Reduce `points_per_side` (fewer SAM proposals) or `n_template_view` in the `SAM6D(...)` constructor.

**`ModuleNotFoundError: No module named 'phoxi_api'`**  
Install it with `pip install phoxi_api`. This is only needed for `sam6d_example_photoneo.ipynb`; the main notebook has no dependency on it.

**`RuntimeError: PhoXi Control is not running`**  
Start PhoXi Control (`/usr/bin/PhoXiControl`) before running the Photoneo notebook. Alternatively, export your scan as TIF/PNG files and use the fallback path in the notebook.

**Depth looks wrong / no detections**  
Check `DEPTH_SCALE`. For a RealSense camera streaming uint16 depth in mm, use `DEPTH_SCALE=1000.0` (converts the `0.0–0.2` raw range to `0–200` mm). For depth already in mm, use `1.0`.

---

## Credits

- [SAM-6D](https://github.com/JiehongLin/SAM-6D) — Lin et al., CVPR 2024
- [FoundationPose](https://github.com/NVlabs/FoundationPose) — Wen et al., CVPR 2024 (Highlight)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) — Kirillov et al., ICCV 2023
- [DINOv2](https://github.com/facebookresearch/dinov2) — Oquab et al., TMLR 2024
