# Real-time Dangerous Behavior Analysis System (MMPose Version)

This project uses YOLOv8 and MMPose (HRNet) to analyze video streams in real-time and detect dangerous behaviors on playground equipment.

---

## Table of Contents
- [Overview](#overview)
- [Environment Preparation](#environment-preparation)
- [Install PyTorch](#install-pytorch)
- [Install MMLab Core Libraries](#install-mmlab-core-libraries)
- [Install Remaining Dependencies](#install-remaining-dependencies)
- [Run the Project](#run-the-project)
- [Troubleshooting & Issues](#troubleshooting--issues)

---

## Overview

> **Important:** This project depends on specific versions of PyTorch, CUDA, and MMLab libraries. **DO NOT** use `pip install mmpose` directly. Follow the steps below using Conda for full compatibility.

---

## Environment Preparation

Create a new Conda environment with Python 3.10:

```powershell
# Create an environment named mmpose_env using Python 3.10
conda create -n mmpose_env python=3.10 -y

# Activate the new environment
conda activate mmpose_env
```

---

## Install PyTorch

Install MMLab-compatible PyTorch 2.1.0 (CUDA 12.1):

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

Verify PyTorch installation (optional):

```bash
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

You should see PyTorch version 2.1.0+cu121 and CUDA as available.

---

## Install MMLab Core Libraries

Install the following in order:

```bash
# 1. openmim and mmengine
pip install -U openmim
mim install mmengine

# 2. mmcv 2.1.0 (pre-compiled for PyTorch 2.1.0)
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# 3. mmpose 1.3.1
pip install mmpose==1.3.1

# 4. mmdet 3.3.0
pip install mmdet==3.3.0

# 5. ffmpeg
conda install ffmpeg -c conda-forge
```

---

## Install Remaining Dependencies

Save the following as `requirements.txt` and install:

```text
ultralytics
requests
numpy<2.0.0
opencv-python
```

Or install directly:

```bash
pip install ultralytics requests "numpy<2.0.0" opencv-python
```

---

## Run the Project

After installation, run the real-time analysis script:

```bash
python live_analysis_mmpose.py
```

- **First Run:** `main_mmpose.py` will auto-download HRNet models. Be patient.
- **Configure Shared Folder:** Set `SHARED_FOLDER_PATH` in `live_analysis_mmpose.py` to a local shared folder.
- **Configure API URL:** Set `RENDER_API_URL` in `live_analysis_mmpose.py` to your backend API address.

---

## Troubleshooting & Issues

- Double-check model paths (e.g., `last.pt`, `yolov8n.pt`). You may need to update these for your setup.
- If you encounter issues, reset the environment and reinstall:

```powershell
conda deactivate
conda env remove -n mmpose_env -y
conda create -n mmpose_env python=3.10 -y
conda activate mmpose_env
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}')"
pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install mmpose==1.3.1
pip install mmdet==3.3.0
pip install opencv-python
pip install ultralytics
pip install numpy
```

Verify environment:

```python
import torch
import mmcv
import mmpose
import mmdet
from ultralytics import YOLO

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA版本: {torch.version.cuda}")
    print(f"✓ GPU设备: {torch.cuda.get_device_name(0)}")
print(f"✓ MMCV: {mmcv.__version__}")
print(f"✓ MMPose: {mmpose.__version__}")
print(f"✓ MMDet: {mmdet.__version__}")
print(f"✓ Ultralytics: OK")
print("\n环境配置成功！")
```

- Torch version must match your CUDA version. If you have torch issues:

```powershell
conda activate mmpose_env
pip list | findstr torch
python -c "import sys; print('\n'.join(sys.path))"
pip uninstall torch torchvision torchaudio -y
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

---

If you need more examples, troubleshooting tips, or want a printable checklist, let me know!

