"""Run with: modal run gpu_check.py"""

import subprocess

import modal

app = modal.App("myssp-gpu-check")


@app.function(gpu="T4", timeout=60)
def check():
    subprocess.run(["nvidia-smi"], check=True)


@app.local_entrypoint()
def main():
    check.remote()
