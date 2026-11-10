# Local files → cloud GPU → local results

Use these examples from the repository folder, with your venv or Conda environment active and Modal credentials configured as in the main README. Install local dependencies with `python -m pip install -r requirements.txt`.

| Script | Purpose |
| --- | --- |
| `prepare_demo_data.py` | Download one public sample photograph to your laptop. |
| `image_pipeline.py` | Upload images, run pretrained GPU inference, and automatically download results. |
| `cloud_files.py` | List, upload, download, edit, and delete persistent cloud files. |

The scripts use a separate Modal Volume named **`myssp-file-demo`**. A Volume is persistent cloud storage, independent of the lifetime of a GPU container. These commands use your configured team token; browser login is not needed for each operation.

## 1. Run the complete image demo

```bash
# Download the public dog photograph from PyTorch's tutorial.
python prepare_demo_data.py

# Upload it, classify it on a T4, and download the saved outputs.
modal run image_pipeline.py --input-dir datasets/demo
```

The second command automatically creates the Volume if needed. It prints a unique run ID and the local output folder, for example:

```text
outputs/20261111T031500Z-a1b2c3d4/
  results.json       # Predictions, confidence scores, and original filenames
  notes.txt          # A small text file for practising cloud edits
  images/
    0001.jpg         # Preview with the predicted label and confidence
```

The model is pretrained **ResNet18**, which classifies the whole image into ImageNet categories. It does not draw object-detection boxes. Public model weights require no Hugging Face token. The first run takes longer to build the container and cache the weights.

To use your own JPG/PNG folder:

```bash
modal run image_pipeline.py --input-dir ./datasets/my-images --max-images 20
```

By default the demo processes at most **8 images**, one at a time; `--max-images` changes that limit. Each selected image must be at most 20 MB. Invalid image files are recorded in `results.json`. The local machine needs only Modal; PyTorch, Torchvision, and Pillow are installed in the cloud image.

### Using `.env` credentials

If you use `.env` instead of `modal token set`, install `python-dotenv[cli]` as described in the README, then prefix commands that contact Modal:

```bash
dotenv -f .env run -- modal run image_pipeline.py --input-dir datasets/demo
dotenv -f .env run -- python cloud_files.py list
```

The `.env` file itself is not uploaded. `prepare_demo_data.py` does not use Modal credentials.

## 2. Understand where the files live

Every image run has its own directory inside the Volume:

```text
myssp-file-demo (cloud Volume)
  models/                          # Reusable pretrained weights
  runs/RUN_ID/
    inputs/                        # Uploaded images and filename manifest
    outputs/
      results.json
      notes.txt
      images/
```

Inside the GPU function, this Volume is mounted at `/data`; for example, cloud `runs/RUN_ID/outputs` is `/data/runs/RUN_ID/outputs` in Python. The file commands below use paths relative to the Volume root, without `/data`.

Cloud results remain after the run. Each run prints a recovery command if the automatic download is interrupted. Replace `RUN_ID` in these examples with the actual ID printed by your run:

```bash
python cloud_files.py list runs/RUN_ID/outputs
python cloud_files.py download runs/RUN_ID/outputs outputs/recovered
```

The download destination is a local **directory**. This example saves files under `outputs/recovered/outputs/`; the remote folder's final name is preserved. Downloads refuse to overwrite existing files unless you add `--overwrite`.

## 3. Practise uploading, editing, and downloading

If you have not run the image demo, create the Volume once:

```bash
python cloud_files.py init
```

Use your own name instead of `my-demo` below so teammates can practise separately. Upload the included disposable text file:

```bash
python cloud_files.py upload sample_data/notes.txt scratch/my-demo/notes.txt
python cloud_files.py list scratch/my-demo
```

Edit the file **in cloud storage** by replacing text:

```bash
python cloud_files.py edit scratch/my-demo/notes.txt --old draft --new reviewed
```

This downloads the existing UTF-8 text file to a temporary location, replaces the matching text, and uploads it back to the same path. It fails if the original text is not found. Changes affect the shared cloud file, so avoid editing the same file simultaneously with a teammate.

Download the edited file:

```bash
python cloud_files.py download scratch/my-demo/notes.txt outputs/edited
```

Open `outputs/edited/notes.txt` locally to see `status: reviewed`.

For larger edits, download the file, edit it in your usual editor, then upload it with explicit overwrite:

```bash
python cloud_files.py upload outputs/edited/notes.txt scratch/my-demo/notes.txt --overwrite
```

You can also upload/download an entire dataset folder using the same commands. Upload only the intended data folder; the file tool rejects common credential filenames and symlinks. Review file contents and archives yourself for secrets; `.gitignore` alone does not control cloud uploads.

## 4. Delete cloud files

Delete the disposable practice file:

```bash
python cloud_files.py delete scratch/my-demo/notes.txt
```

Delete an entire completed image run, including its uploaded inputs and outputs:

```bash
python cloud_files.py delete runs/RUN_ID --recursive
```

The script displays the exact Volume and path and asks for confirmation. Directory deletion requires `--recursive`; `--yes` skips the confirmation for deliberate automation. Deletion is permanent in the cloud and does not delete copies already downloaded to your laptop. Deleting the Volume root is blocked.

## 5. Useful variations

```bash
# See all commands and options.
python cloud_files.py --help

# Work with the original text demo's Volume instead.
python cloud_files.py --volume myssp-demo-data list outputs
```

Put `--volume NAME` **before** the subcommand. File operations use the Modal storage API/CLI and do not start a GPU, though storage/network charges can apply. Only `image_pipeline.py` requests GPU compute. The scripts do not erase your cloud results automatically.

To adapt the GPU work, replace the model and processing loop in `classify_images()` while keeping the upload, `volume.commit()`, and download steps. Save additional outputs beneath that run's `outputs` directory and include their relative filenames in the returned list so they are downloaded too.

Local tests use simulated cloud calls and do not start GPUs:

```bash
python -m unittest discover -s tests
```

References: [Modal Volumes](https://modal.com/docs/guide/volumes), [file commands](https://modal.com/docs/cli/latest/volume), [ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), and the [sample photograph's source tutorial](https://pytorch.org/hub/pytorch_vision_resnet/).
