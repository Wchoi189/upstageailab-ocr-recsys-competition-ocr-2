import os

from PIL import Image


def convert_png_to_webp(root_directory):
    for root, dirs, files in os.walk(root_directory):
        for filename in files:
            if filename.endswith(".png"):
                png_path = os.path.join(root, filename)
                webp_path = os.path.join(root, filename.replace(".png", ".webp"))

                try:
                    with Image.open(png_path) as img:
                        img.save(webp_path, "WEBP", quality=80)
                    print(f"Converted {png_path} to {webp_path}")
                except Exception as e:
                    print(f"Failed to convert {png_path}: {e}")


if __name__ == "__main__":
    assets_dir = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/assets/images/"
    convert_png_to_webp(assets_dir)
