from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
image_path = ROOT / "examples" / "sample_image.png"
mask_path = ROOT / "examples" / "sample_mask.png"
image_path.parent.mkdir(parents=True, exist_ok=True)

image = Image.new("RGB", (768, 512), (185, 215, 235))
draw = ImageDraw.Draw(image)
draw.rectangle((0, 330, 768, 512), fill=(90, 145, 75))
draw.ellipse((560, 35, 690, 165), fill=(250, 220, 95))
draw.rectangle((100, 190, 260, 350), fill=(190, 75, 50))
draw.polygon([(75, 190), (180, 100), (285, 190)], fill=(110, 55, 45))
draw.rectangle((145, 260, 205, 350), fill=(75, 55, 40))
draw.rectangle((420, 270, 480, 390), fill=(55, 95, 150))
draw.ellipse((390, 220, 510, 330), fill=(55, 95, 150))
draw.line((450, 330, 425, 420), fill=(30, 45, 70), width=12)
draw.line((450, 330, 480, 420), fill=(30, 45, 70), width=12)
image.save(image_path)

mask = Image.new("L", image.size, 0)
mask_draw = ImageDraw.Draw(mask)
mask_draw.rectangle((390, 205, 515, 345), fill=255)
mask.save(mask_path)
print(f"Created {image_path}")
print(f"Created {mask_path} (white=reconstruct, black=preserve)")

