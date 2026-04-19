"""
Copies rendered test views into submission/ with correct filenames
matching the held-out input image names (LLFF hold=8 split).
"""
import os, shutil, glob

ALL_SCENES = ["aeroplane", "bike", "buddha", "cycle", "face", "firehydrant", "still3", "toy"]

for scene in ALL_SCENES:
    # Get test image names from raw (every 8th image = LLFF hold=8)
    raw_imgs = sorted(glob.glob(f"data/raw/competition/{scene}/images/*.JPG") +
                      glob.glob(f"data/raw/competition/{scene}/images/*.jpg") +
                      glob.glob(f"data/raw/competition/{scene}/images/*.png"))
    test_imgs = [p for i, p in enumerate(raw_imgs) if i % 8 == 0]

    # Get rendered outputs
    render_dir = f"output/{scene}_sr/test/ours_30000/renders"
    renders = sorted(glob.glob(f"{render_dir}/*.png") + glob.glob(f"{render_dir}/*.jpg"))

    out_dir = f"submission/{scene}"
    os.makedirs(out_dir, exist_ok=True)

    if len(renders) != len(test_imgs):
        print(f"WARNING {scene}: {len(renders)} renders vs {len(test_imgs)} test imgs")

    for render, test_path in zip(renders, test_imgs):
        test_fname = os.path.basename(test_path)
        # Keep original extension from test image name
        name, _ = os.path.splitext(test_fname)
        out_fname = test_fname  # exact match required by competition
        shutil.copy(render, os.path.join(out_dir, out_fname))
        print(f"  {scene}/{out_fname}")

    print(f"[{scene}] {len(renders)} test views -> submission/{scene}/")

print("\nDone! Now run: python imgs2csv.py")
