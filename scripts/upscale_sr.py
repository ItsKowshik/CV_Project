"""
RealESRGAN 4x upscaler for post-render SR.
Usage: python scripts/upscale_sr.py <scene1> <scene2> ...
       OR called internally by full_pipeline.sh
"""
import os, sys, glob
from PIL import Image
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def get_images(directory):
    """Glob all common image extensions, case-insensitive."""
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(directory, ext))
    return sorted(set(files))

def build_upsampler(model_path='models/RealESRGAN_x4plus.pth'):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(scale=4, model_path=model_path, model=model,
                        tile=256, tile_pad=10, pre_pad=0, half=True)

if __name__ == '__main__':
    scenes = sys.argv[1:] if len(sys.argv) > 1 else []
    if not scenes:
        print("Usage: python scripts/upscale_sr.py <scene1> <scene2> ...")
        sys.exit(1)

    upsampler = build_upsampler()

    for scene in scenes:
        in_dir  = f'data/raw/competition/{scene}/images'
        out_dir = f'data/processed/{scene}/images'
        os.makedirs(out_dir, exist_ok=True)

        imgs = get_images(in_dir)
        if not imgs:
            print(f'[{scene}] ERROR: No images found in {in_dir}'); continue

        print(f'[{scene}] Upscaling {len(imgs)} images -> {out_dir}')
        for path in imgs:
            out_path = os.path.join(out_dir, os.path.basename(path))
            if os.path.exists(out_path):
                continue  # resumable
            img = np.array(Image.open(path).convert('RGB'))
            out, _ = upsampler.enhance(img, outscale=4)
            Image.fromarray(out).save(out_path)
            print(f'  {os.path.basename(path)}', flush=True)
        print(f'[{scene}] Done')
