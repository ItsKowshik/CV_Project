#!/bin/bash
set -e
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

SCENES_WITH_SPARSE="bike buddha firehydrant toy"
SCENES_NO_SPARSE="aeroplane cycle face still3"
ALL_SCENES="aeroplane bike buddha cycle face firehydrant still3 toy"

echo "=== Step 1: COLMAP SfM for scenes without sparse/ ==="
for scene in $SCENES_NO_SPARSE; do
  if [ -d "data/raw/competition/$scene/sparse/0" ]; then
    echo "  $scene: sparse already exists, skipping"
    continue
  fi
  echo "  Running COLMAP for $scene..."
  DB=data/raw/competition/$scene/database.db
  IMAGES=data/raw/competition/$scene/images
  SPARSE=data/raw/competition/$scene/sparse
  mkdir -p $SPARSE
  colmap feature_extractor --database_path $DB --image_path $IMAGES \
    --ImageReader.single_camera 1 --SiftExtraction.use_gpu 1
  colmap exhaustive_matcher --database_path $DB --SiftMatching.use_gpu 1
  colmap mapper --database_path $DB --image_path $IMAGES --output_path $SPARSE
  echo "  $scene: COLMAP done"
done

echo ""
echo "=== Step 2: RealESRGAN 4x upscaling ==="
for scene in $ALL_SCENES; do
  if [ -d "data/processed/$scene/images" ] && [ "$(ls -A data/processed/$scene/images)" ]; then
    echo "  $scene: already upscaled, skipping"
    continue
  fi
  echo "  Upscaling $scene..."
  mkdir -p data/processed/$scene/images
  cp -r data/raw/competition/$scene/sparse data/processed/$scene/sparse 2>/dev/null || true
  python scripts/upscale_sr.py $scene
done

echo ""
echo "=== Step 3: Train 3DGS on HR images (30k iterations) ==="
for scene in $ALL_SCENES; do
  if [ -d "output/${scene}_sr/point_cloud" ]; then
    echo "  $scene: already trained, skipping"
    continue
  fi
  echo "  Training $scene..."
  python src/gaussian_splatting/train.py \
    -s data/processed/$scene \
    -m output/${scene}_sr \
    --iterations 30000 \
    --eval \
    --ip 127.0.0.1 --port 0
  echo "  $scene: training done"
done

echo ""
echo "=== Step 4: Render test views ==="
for scene in $ALL_SCENES; do
  echo "  Rendering $scene..."
  python src/gaussian_splatting/render.py \
    -m output/${scene}_sr \
    --skip_train
done

echo ""
echo "=== Step 5: Organize submission ==="
python scripts/prepare_submission.py

echo ""
echo "=== Pipeline complete! ==="
echo "Run: python imgs2csv.py  to generate submission.csv"
