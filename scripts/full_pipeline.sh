#!/bin/bash
# ============================================================
# Full 3DGS Super-Resolution Pipeline — EE5178
# Usage: tmux new -s pipeline then bash scripts/full_pipeline.sh
# Fully resumable: re-run safely if interrupted
# ============================================================
set -e
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

SCENES_NO_SPARSE="aeroplane cycle face still3"
ALL_SCENES="aeroplane bike buddha cycle face firehydrant still3 toy"
ITERS=30000
mkdir -p logs submission
LOG="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a $LOG) 2>&1  # log everything

log() { echo "[$(date '+%H:%M:%S')] $1"; }

get_images() {
  # Returns all image files regardless of case
  find "$1" -maxdepth 1 \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort
}

# ── Stage 0: COLMAP for scenes missing sparse/ ─────────────
log "=== STAGE 0: COLMAP SfM ==="
for scene in $SCENES_NO_SPARSE; do
  if [ -d "data/raw/competition/$scene/sparse/0" ]; then
    log "  $scene: sparse exists, skipping"; continue
  fi
  log "  Running COLMAP for $scene (bicubic 8x upscale for feature detection)..."

  # Bicubic upscale for COLMAP only (better SIFT on tiny images)
  python - << PYEOF
import os, glob
from PIL import Image
scene = "$scene"
raw = sorted(
    glob.glob(f"data/raw/competition/{scene}/images/*.jpg") +
    glob.glob(f"data/raw/competition/{scene}/images/*.jpeg") +
    glob.glob(f"data/raw/competition/{scene}/images/*.png") +
    glob.glob(f"data/raw/competition/{scene}/images/*.JPG")
)
out_dir = f"data/colmap_input/{scene}/images"
os.makedirs(out_dir, exist_ok=True)
for p in raw:
    img = Image.open(p)
    w, h = img.size
    img.resize((w*8, h*8), Image.BICUBIC).save(
        os.path.join(out_dir, os.path.basename(p)))
print(f"{scene}: {len(raw)} images upscaled for COLMAP")
PYEOF

  DB=data/colmap_input/$scene/database.db
  IMGS=data/colmap_input/$scene/images
  SP=data/raw/competition/$scene/sparse
  rm -f $DB; rm -rf $SP; mkdir -p $SP

  colmap feature_extractor \
    --database_path $DB --image_path $IMGS \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.max_num_features 8192

  colmap exhaustive_matcher \
    --database_path $DB \
    --SiftMatching.use_gpu 1 \
    --SiftMatching.guided_matching 1

  colmap mapper \
    --database_path $DB --image_path $IMGS \
    --output_path $SP \
    --Mapper.init_min_tri_angle 4 \
    --Mapper.multiple_models 0

  if [ -d "$SP/0" ]; then
    log "  $scene: COLMAP OK"
  else
    log "  WARNING: $scene COLMAP failed — will skip in training"
  fi
done

# ── Stage 1: Train 3DGS on LR images ───────────────────────
log "=== STAGE 1: 3DGS Training ($ITERS iters) ==="
for scene in $ALL_SCENES; do
  # Skip if no sparse reconstruction available
  if [ ! -d "data/raw/competition/$scene/sparse/0" ]; then
    log "  $scene: no sparse/, skipping training"; continue
  fi
  if [ -d "output/${scene}_lr/point_cloud/iteration_${ITERS}" ]; then
    log "  $scene: already trained, skipping"; continue
  fi
  log "  Training $scene..."
  python src/gaussian_splatting/train.py \
    -s data/raw/competition/$scene \
    -m output/${scene}_lr \
    --iterations $ITERS \
    --eval \
    --ip 127.0.0.1 --port 0
  log "  $scene: training complete"
done

# ── Stage 2: Render test views ─────────────────────────────
log "=== STAGE 2: Rendering test views ==="
for scene in $ALL_SCENES; do
  if [ ! -d "output/${scene}_lr" ]; then
    log "  $scene: no trained model, skipping"; continue
  fi
  if [ -d "output/${scene}_lr/test" ]; then
    log "  $scene: renders exist, skipping"; continue
  fi
  log "  Rendering $scene..."
  python src/gaussian_splatting/render.py \
    -m output/${scene}_lr --skip_train
  log "  $scene: rendering done"
done

# ── Stage 3: Post-render SR upscaling ──────────────────────
log "=== STAGE 3: Post-render RealESRGAN 4x SR ==="
python - << 'PYEOF'
import os, glob
from PIL import Image
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def get_images(d):
    files = []
    for ext in ['*.jpg','*.jpeg','*.png','*.JPG','*.JPEG','*.PNG']:
        files += glob.glob(os.path.join(d, ext))
    return sorted(set(files))

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(scale=4, model_path='models/RealESRGAN_x4plus.pth',
                          model=model, tile=256, tile_pad=10,
                          pre_pad=0, half=True)

ALL_SCENES = ["aeroplane","bike","buddha","cycle","face","firehydrant","still3","toy"]
ITERS = 30000

for scene in ALL_SCENES:
    # Get held-out test filenames (LLFF hold=8: every 8th image)
    raw_dir = f"data/raw/competition/{scene}/images"
    raw_imgs = get_images(raw_dir)
    test_names = [os.path.basename(p) for i, p in enumerate(raw_imgs) if i % 8 == 0]

    # Find rendered outputs
    render_dirs = sorted(glob.glob(f"output/{scene}_lr/test/ours_*/renders"))
    if not render_dirs:
        print(f"[{scene}] No renders found, skipping"); continue
    renders = get_images(render_dirs[-1])

    out_dir = f"submission/{scene}"
    os.makedirs(out_dir, exist_ok=True)

    if len(renders) != len(test_names):
        print(f"[{scene}] WARNING: {len(renders)} renders vs {len(test_names)} test names — zipping by order")

    print(f"[{scene}] SR upscaling {len(renders)} renders...")
    for render_path, test_name in zip(renders, test_names):
        out_path = os.path.join(out_dir, test_name)
        if os.path.exists(out_path):
            continue  # resumable
        img = np.array(Image.open(render_path).convert('RGB'))
        out, _ = upsampler.enhance(img, outscale=4)
        Image.fromarray(out).save(out_path)
        print(f"  {test_name}", flush=True)
    print(f"[{scene}] -> submission/{scene}/ ({len(renders)} images)")
PYEOF

# ── Stage 4: Generate submission CSV ───────────────────────
log "=== STAGE 4: Generating submission.csv ==="
[ -f "imgs2csv.py" ] || find data/ scripts/ -name "imgs2csv.py" -exec cp {} . \;
python imgs2csv.py

log "=== PIPELINE COMPLETE ==="
log "Check: submission.csv"
log "Full log: $LOG"
