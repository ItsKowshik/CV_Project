import os, sys, glob
from PIL import Image
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(scale=4, model_path='models/RealESRGAN_x4plus.pth',
                          model=model, tile=512, tile_pad=10,
                          pre_pad=0, half=True)

scenes = sys.argv[1:]
for scene in scenes:
    in_dir  = f'data/raw/competition/{scene}/images'
    out_dir = f'data/processed/{scene}/images'
    os.makedirs(out_dir, exist_ok=True)
    imgs = sorted(glob.glob(f'{in_dir}/*.JPG') + glob.glob(f'{in_dir}/*.jpg') + glob.glob(f'{in_dir}/*.png'))
    print(f'[{scene}] Upscaling {len(imgs)} images...')
    for path in imgs:
        img = np.array(Image.open(path).convert('RGB'))
        out, _ = upsampler.enhance(img, outscale=4)
        fname = os.path.basename(path)
        Image.fromarray(out).save(os.path.join(out_dir, fname))
        print(f'  {fname}', flush=True)
    print(f'[{scene}] Done -> {out_dir}')
