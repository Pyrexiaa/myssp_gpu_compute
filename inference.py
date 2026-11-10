"""Load pretrained weights and classify one English sentence on a cloud GPU."""

import json
from pathlib import Path
from uuid import uuid4

import modal

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
VOLUME_NAME = "myssp-demo-data"

app = modal.App("myssp-inference-demo")

# 1. Install research dependencies in the cloud, not on your laptop.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "transformers==4.57.1")
    .env({"HF_HUB_CACHE": "/data/models"})
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)


@app.function(image=image, gpu="T4", volumes={"/data": volume}, timeout=600)
def predict(text: str) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # 2. Download/cache the pretrained weights and load the model onto the GPU.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, use_safetensors=True
    ).to("cuda")
    model.eval()
    volume.commit()

    # 3. Move the input to the same GPU and run inference without training.
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to("cuda")
    with torch.inference_mode():
        probabilities = model(**inputs).logits.softmax(dim=-1)[0]
    label_id = probabilities.argmax().item()
    result = {
        "model": MODEL_ID,
        "text": text,
        "label": model.config.id2label[label_id],
        "score": probabilities[label_id].item(),
    }

    # 4. Save and commit a unique result so it survives this container stopping.
    result_path = Path("/data/outputs") / f"{uuid4().hex}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    volume.commit()
    return {**result, "volume_path": str(result_path.relative_to("/data"))}


@app.local_entrypoint()
def main(text: str = "I really enjoyed this movie!"):
    if not text.strip():
        raise ValueError("Please provide a non-empty sentence with --text.")
    result = predict.remote(text)
    print(f"{result['label']} (score: {result['score']:.4f})")
    print("Download this result with:")
    print(
        f"modal volume get {VOLUME_NAME} {result['volume_path']} "
        f"result-{Path(result['volume_path']).name}"
    )
