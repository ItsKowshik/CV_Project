# 3D Gaussian Splatting Super-Resolution — EE5178 Competition

Reconstruct high-resolution novel views from low-resolution multi-view images using 3DGS + RealESRGAN 4× upscaling.

## Requirements

- Ubuntu 22.04
- NVIDIA GPU with CUDA 12.8 support (tested on RTX 5050)
- micromamba / conda environment with Python 3.11
- PyTorch 2.11.0+cu128

## Setup (one-time)

```bash
git clone --recursive https://github.com/YOUR_USERNAME/3dgs-superres.git
cd 3dgs-superres




Create conda env
micromamba create -n main python=3.11 -y
micromamba activate main

Install PyTorch with CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Run full setup (CUDA extensions, deps, COLMAP, model weights)
bash scripts/setup_env.sh

text

## Dataset

Download `ee-5178-3-d-gaussian-splatting-super-resolution-challenge.zip` from the competition page and place as:
data/raw/competition/
├── aeroplane/images/
├── bike/images/, sparse/
├── buddha/images/, sparse/
├── cycle/images/
├── face/images/
├── firehydrant/images/, sparse/
├── still3/images/
└── toy/images/, sparse/

text

## Run Full Pipeline

```bash
bash scripts/run_pipeline.sh
```

This will:
1. Run COLMAP SfM on scenes without sparse/ (aeroplane, cycle, face, still3)
2. Upscale all images 4× with RealESRGAN
3. Train 3DGS for 30k iterations per scene
4. Render held-out test views
5. Organize outputs into `submission/`

Then generate the submission CSV:
```bash
python imgs2csv.py
```

## Score Formula
Score = 0.5 × SSIM + 0.5 × (PSNR / 30)

text

## Baseline Results (7k iters, no SR)

| Scene | PSNR (test) |
|-------|------------|
| bike  | 18.68 dB   |
EOF