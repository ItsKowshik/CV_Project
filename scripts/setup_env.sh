#!/bin/bash
set -e
echo "=== 3DGS Super-Resolution Setup ==="

# 1. Install CUDA 12.8 toolkit (required to build CUDA extensions)
echo "[1/5] Checking CUDA 12.8..."
if ! nvcc --version 2>/dev/null | grep -q "12.8"; then
  echo "Installing CUDA 12.8 toolkit..."
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt update -q
  sudo apt install -y cuda-toolkit-12-8
  echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
  echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
  source ~/.bashrc
fi

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

# 2. Install Python deps
echo "[2/5] Installing Python dependencies..."
pip install ninja plyfile tqdm einops timm lpips scikit-image torchmetrics \
            wandb omegaconf PyYAML matplotlib imageio imageio-ffmpeg \
            opencv-python basicsr realesrgan

# Fix basicsr torchvision compatibility
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
  $(python -c "import basicsr; import os; print(os.path.dirname(basicsr.__file__))")/data/degradations.py
echo "  basicsr patched OK"

# 3. Build CUDA submodule extensions
echo "[3/5] Building CUDA extensions..."
for module in diff-gaussian-rasterization simple-knn fused-ssim; do
  MOD_PATH="src/gaussian_splatting/submodules/$module"
  # patch cstdint if needed
  find $MOD_PATH -name "*.h" -exec grep -l "uint32_t\|uintptr_t" {} \; | while read f; do
    grep -q "cstdint" "$f" || sed -i '1s/^/#include stdint>\n/' "$f"
  done
  pip install --no-build-isolation $MOD_PATH/
  echo "  $module built OK"
done

# 4. Install COLMAP (for SfM on scenes without sparse/)
echo "[4/5] Installing COLMAP..."
micromamba install -c conda-forge colmap -y -q 2>/dev/null || \
  conda install -c conda-forge colmap -y -q 2>/dev/null || \
  echo "  WARNING: Install COLMAP manually: micromamba install -c conda-forge colmap"

# 5. Download RealESRGAN weights
echo "[5/5] Downloading RealESRGAN weights..."
mkdir -p models
wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
  -O models/RealESRGAN_x4plus.pth
echo "  Weights saved to models/RealESRGAN_x4plus.pth"

echo ""
echo "=== Setup complete! ==="
echo "Next: Place competition dataset at data/raw/competition/ then run:"
echo "  bash scripts/run_pipeline.sh"
