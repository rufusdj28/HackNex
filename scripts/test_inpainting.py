import os
from pathlib import Path

import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama


ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "examples" / "sample_image.png"
MASK_PATH = ROOT / "examples" / "sample_mask.png"
OUTPUT_PATH = ROOT / "data" / "output" / "sample_reconstructed.png"


def main() -> None:
    if not IMAGE_PATH.exists() or not MASK_PATH.exists():
        raise FileNotFoundError("Run scripts/create_sample.py first.")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime: {torch.version.cuda}")

    # simple-lama-inpainting uses the available torch device and downloads one
    # pretrained LaMa checkpoint on first construction.
    os.environ.setdefault("TORCH_HOME", str(ROOT / "models" / "torch"))
    lama = SimpleLama()
    image = Image.open(IMAGE_PATH).convert("RGB")
    mask = Image.open(MASK_PATH).convert("L")
    result = lama(image, mask)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT_PATH)
    print(f"Saved reconstructed image: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

