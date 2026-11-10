"""Upload images, classify them on a T4, and download JSON and labelled previews."""

from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
from uuid import uuid4

import modal

VOLUME_NAME = "myssp-file-demo"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
app = modal.App("myssp-image-file-demo")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", "pillow==11.3.0")
    .env({"TORCH_HOME": "/data/models"})
)


def select_images(input_dir: str, max_images: int) -> list[Path]:
    """Select image files only; never send a whole project directory to the cloud."""
    root = Path(input_dir).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("--input-dir must be an existing folder, not a symlink.")
    if max_images < 1:
        raise ValueError("--max-images must be at least 1.")
    selected = []
    for directory, folders, files in os.walk(root, followlinks=False):
        # Hidden files, credentials, and project folders are never selected.
        for name in folders + files:
            if (Path(directory) / name).is_symlink():
                raise ValueError("Dataset contains a symlink; copy the actual images instead.")
        folders[:] = sorted(name for name in folders if not name.startswith("."))
        for name in sorted(files):
            file = Path(directory) / name
            if name.startswith(".") or file.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if file.stat().st_size > 20 * 1024 * 1024:
                raise ValueError(f"Image exceeds the 20 MB demo limit: {file}")
            selected.append(file)
    if not selected:
        raise ValueError("No JPG/PNG images found. Run python prepare_demo_data.py first.")
    if len(selected) > max_images:
        print(f"Using {max_images} of {len(selected)} images; change --max-images to process more.")
    return selected[:max_images]


def upload_images(files: list[Path], input_dir: str, run_id: str):
    """Keep local names in a manifest; use predictable, unique cloud filenames."""
    root = Path(input_dir).expanduser()
    manifest = [
        {"source": str(file.relative_to(root)), "stored": f"{i:04d}{file.suffix.lower()}"}
        for i, file in enumerate(files, 1)
    ]
    with BytesIO(json.dumps(manifest).encode("utf-8")) as metadata:
        with volume.batch_upload() as batch:
            for file, item in zip(files, manifest):
                batch.put_file(file, f"/runs/{run_id}/inputs/{item['stored']}")
            batch.put_file(metadata, f"/runs/{run_id}/inputs/manifest.json")


@app.function(image=image, gpu="T4", volumes={"/data": volume}, timeout=900)
def classify_images(run_id: str) -> list[str]:
    """This function runs in the cloud; only this step requests a GPU."""
    import torch
    from torchvision.models import ResNet18_Weights, resnet18
    from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}", run_id):
        raise ValueError("Invalid run ID.")
    volume.reload()  # Also see fresh uploads if Modal reuses an existing container.
    run = Path("/data/runs") / run_id
    manifest = json.loads((run / "inputs/manifest.json").read_text())
    output = run / "outputs"
    (output / "images").mkdir(parents=True, exist_ok=False)

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).to("cuda").eval()
    preprocess = weights.transforms()
    labels = weights.meta["categories"]
    volume.commit()  # Persist downloaded model weights for later runs.
    print(f"Running {len(manifest)} images on {torch.cuda.get_device_name(0)}")

    predictions = []
    output_files = []
    for i, item in enumerate(manifest, 1):
        stored = item["stored"]
        if not re.fullmatch(r"[0-9]+\.(jpg|jpeg|png)", stored):
            raise ValueError("Invalid image filename in uploaded manifest.")
        try:
            with Image.open(run / "inputs" / stored) as original:
                photo = ImageOps.exif_transpose(original).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            predictions.append({"source": item["source"], "error": type(error).__name__})
            continue
        with torch.inference_mode():
            scores = model(preprocess(photo).unsqueeze(0).to("cuda")).softmax(dim=1)[0]
        score, label_id = scores.max(dim=0)
        label = labels[label_id.item()]
        score = score.item()

        # Save a small preview with the predicted class and confidence.
        photo.thumbnail((640, 480))
        preview = Image.new("RGB", (max(480, photo.width), photo.height + 64), "#152030")
        preview.paste(photo, (0, 0))
        ImageDraw.Draw(preview).text(
            (10, photo.height + 12), f"{label} ({score:.1%})",
            fill="white", font=ImageFont.load_default(size=20),
        )
        name = f"images/{i:04d}.jpg"
        preview.save(output / name, quality=90)
        output_files.append(name)
        predictions.append({"source": item["source"], "label": label, "score": score, "image": name})

    result = {"run_id": run_id, "model": "resnet18/IMAGENET1K_V1", "predictions": predictions}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "notes.txt").write_text("status: draft\n", encoding="utf-8")
    volume.commit()  # Close/write files first, then make them available to the laptop.
    print(f"Saved {len(output_files)} previews; {len(predictions) - len(output_files)} image decode errors.")
    return ["results.json", "notes.txt", *output_files]


def download_outputs(run_id: str, filenames: list[str], destination: Path):
    """Stream only this run's outputs; do not overwrite existing local files."""
    relative_paths = []
    for name in filenames:
        relative = PurePosixPath(name)
        if (
            not relative.parts or relative.is_absolute() or ".." in relative.parts
            or "\\" in name or ":" in name or any(ord(char) < 32 for char in name)
            or str(relative) != name or any(part.startswith(".") for part in relative.parts)
        ):
            raise ValueError("Unexpected output path from the cloud.")
        if relative in relative_paths:
            raise ValueError("Duplicate output path from the cloud.")
        relative_paths.append(relative)
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    for relative in relative_paths:
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.resolve().is_relative_to(root):
            raise ValueError("Output path escapes the local results folder.")
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.part")
        created_temporary = False
        try:
            with temporary.open("xb") as local:
                created_temporary = True
                for chunk in volume.read_file(f"/runs/{run_id}/outputs/{relative}"):
                    local.write(chunk)
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Refusing to overwrite {target}")
            temporary.rename(target)
        except BaseException:
            if created_temporary:
                temporary.unlink(missing_ok=True)
            raise


@app.local_entrypoint()
def main(input_dir: str = "datasets/demo", output_dir: str = "outputs", max_images: int = 8):
    """This entrypoint runs on your laptop and coordinates upload/run/download."""
    files = select_images(input_dir, max_images)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    destination = Path(output_dir).expanduser() / run_id
    print(f"Volume: {VOLUME_NAME}; run: {run_id}")
    print(f"1/3 Uploading {len(files)} images to runs/{run_id}/inputs")
    upload_images(files, input_dir, run_id)
    print("2/3 Running pretrained image classification on the cloud GPU")
    outputs = classify_images.remote(run_id)
    print("3/3 Downloading predictions and labelled images")
    print(f"If download is interrupted, recover with: python cloud_files.py download runs/{run_id}/outputs outputs/recovered-{run_id}")
    download_outputs(run_id, outputs, destination)
    print(f"Done. Local results: {destination.resolve()}")
    print(f"Cloud files remain at runs/{run_id}; use cloud_files.py to list, edit, or delete them.")
