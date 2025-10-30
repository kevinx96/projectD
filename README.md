# Real-time Dangerous Behavior Analysis System (MMPose Version)
This project uses YOLOv8 and MMPose(HRNet) to analyze video streams in real-time and detect dangerous behaviors on playground equipment.
As this project depends on specific versions of PyTorch, CUDA, and MMLab libraries, DO NOT attempt to simply run pip install mmpose. Please strictly follow the steps below to install in a Conda environment to ensure 100% compatibility of all dependencies.
1. Environment Preparation: Create Conda Environment
First, create a new Conda environment. We will use Python 3.10, as it is the version we have successfully verified.
# Create an environment named mmpose_env using Python 3.10
conda create -n mmpose_env python=3.10 -y

# Activate the new environment
conda activate mmpose_env


2. Install PyTorch (Stable Version)
We will install MMLab-compatible PyTorch 2.1.0 (corresponding to CUDA 12.1). Using pip and specifying the official download source is more stable than using the conda command.
# Install PyTorch 2.1.0, torchvision 0.16.0, and torchaudio 2.1.0 (all corresponding to CUDA 12.1)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

Verify PyTorch Installation (Optional):
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"


You should see the PyTorch version as 2.1.0+cu121 and CUDA as available.
3. Install MMLab Core Libraries (Critical Step)
The MMLab library versions must match exactly, and the installation order is also important.
# 3a. Install openmim and mmengine
pip install -U openmim
mim install mmengine

# 3b. Install mmcv 2.1.0 (pre-compiled version compatible with PyTorch 2.1.0)
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# 3c. Install mmpose 1.3.1 (compatible with mmcv 2.1.0)
pip install mmpose==1.3.1

# 3d. Install mmdet 3.3.0 (compatible with mmcv 2.1.0)
pip install mmdet==3.3.0
# 3e Install ffmpeg
conda install ffmpeg -c conda-forge

4. Install Remaining Dependencies
Finally, we install the remaining Python libraries. You can save the following content as a requirements.txt file and then run pip install -r requirements.txt.
# requirements.txt

# Core AI model
ultralytics

# Network requests
requests

# Ensure NumPy version is compatible with MMLab
numpy<2.0.0

# OpenCV (if not automatically installed by mmpose)
opencv-python




Quick Install Command:
pip install ultralytics requests "numpy<2.0.0" opencv-python 


5. Run the Project
After completing all installation steps, you can run the real-time analysis script.
(First Run) Download Models: The main_mmpose.py script will automatically download the required HRNet models. Please be patient.
Configure Shared Folder: Ensure the SHARED_FOLDER_PATH variable in the live_analysis_mmpose.py script points to a folder you have created and shared locally.
Configure API URL: Ensure the RENDER_API_URL variable in the live_analysis_mmpose.py script points to your deployed backend API address.
Start Real-time Analysis:
python live_analysis_mmpose.py




6.Issues
Double check the path to models including last.pt(this is on main but I as developing new models so the path isn't correct, you need to change it by yourself) and the yolov8n.pt which you can replace with yolov8s or yolov8l, which is for human detecting.
 If still having trouble do the following:
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
echo "========== 环境验证 =========="
python << EOF
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
EOF

P.S. the torch ver should be capable for your own cuda, if not, then sry, I tried decades to find the combination that only fits this project.
try this if its simply the torch missing
conda activate mmpose_env
pip list | grep torch
python -c "import sys; print('\n'.join(sys.path))"
pip uninstall torch torchvision torchaudio -y
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

