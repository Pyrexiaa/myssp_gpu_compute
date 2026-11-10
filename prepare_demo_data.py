"""Download the public dog photograph used in PyTorch's ResNet tutorial."""

import argparse
from pathlib import Path
from urllib.request import urlopen

SOURCE_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="datasets/demo")
    args = parser.parse_args()
    folder = Path(args.output_dir).expanduser()
    target = folder / "dog.jpg"
    if target.exists() or target.is_symlink():
        parser.error(f"{target} already exists; use it or choose another output folder.")
    print("Downloading the public PyTorch example image (no cloud GPU used).")
    with urlopen(SOURCE_URL, timeout=30) as response:
        data = response.read(5 * 1024 * 1024 + 1)
    if len(data) > 5 * 1024 * 1024 or not data.startswith(b"\xff\xd8\xff"):
        raise ValueError("The download was not a JPEG within the 5 MB demo limit.")
    folder.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as file:
        file.write(data)
    print(f"Saved {target}")
    print(f"Source: {SOURCE_URL}")
    print(f'Next: modal run image_pipeline.py --input-dir "{folder}"')


if __name__ == "__main__":
    main()
