# Team GPU quick start

Run Python on a cloud GPU with **Modal**. Start with a GPU check, then load pretrained model weights and run inference. You do not need a local NVIDIA GPU or CUDA installation.

## 1. Set up your computer (once)

Clone this repository and open a terminal in the `myssp_gpu_compute` folder. **Choose one environment option below: venv or Conda.**

### Option A: venv

Use Python 3.11 or newer.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

On macOS/Linux, use `python3` for the first command if `python` is unavailable.

```bash
python -m pip install -r requirements.txt
```

### Option B: Conda

If you already have Miniconda or Anaconda installed, run:

```bash
conda create -n myssp-gpu python=3.11 pip -y
conda activate myssp-gpu
python -m pip install -r requirements.txt
```

On Windows, use Anaconda Prompt or a terminal configured for Conda. If activation reports that your shell is not initialized, initialize your shell (for example, `conda init zsh` for zsh or `conda init powershell` for PowerShell), then reopen the terminal and activate the environment again.

**For either option:** continue with steps 2–4 below in the same terminal with your environment active. Only Modal is installed locally. The inference script installs PyTorch and Transformers in the cloud container.

## 2. Connect to the team workspace (once)

Get the private Modal token ID and secret from the team lead/MRG. Run:

```bash
modal token set --token-id your_token_id --token-secret your_token_secret
modal token info
```

Enter the credentials at the prompts. This keeps the values out of the command you type and your shell history. Check that `token info` shows the intended **MRG workspace** before running a job. Modal saves credentials locally in `~/.modal.toml`; never copy that file into Git.

### Optional: use a local `.env` file

Instead of `modal token set`, you can use the included `.env.example` template:

1. Copy `.env.example` to a file named `.env` in this repository. Keep your existing `.env` if you already have one.
2. Edit **only `.env`** and replace both placeholders with the team's Modal token ID and secret. Leave `.env.example` as placeholders.
3. With your venv or Conda environment active, install the optional loader and check the workspace:

```bash
python -m pip install "python-dotenv[cli]"
dotenv -f .env run -- modal token info
```

After confirming the intended MRG workspace, run the demos:

```bash
dotenv -f .env run -- modal run gpu_check.py
dotenv -f .env run -- modal run inference.py --text "I really enjoyed this movie!"
```

**The demos do not automatically load `.env`.** For this option, prefix every Modal command below with `dotenv -f .env run --`, including in later sessions. The variables apply to that local command and override saved Modal token credentials; they are not automatically passed to the GPU function. `.env` is ignored by Git; only the placeholder `.env.example` should be committed. See [python-dotenv](https://bbc2.github.io/python-dotenv/) and [Modal authentication](https://modal.com/docs/sdk/py/latest/config).

## 3. Check the GPU

```bash
modal run gpu_check.py
```

**Success:** an NVIDIA GPU status table appears in your terminal. The script uses a **T4** GPU in Modal.

## 4. Load pretrained weights and run inference

```bash
modal run inference.py --text 'I really enjoyed this movie!'
```

This demo uses a small public [DistilBERT sentiment model](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english). It labels an English sentence `POSITIVE` or `NEGATIVE` and returns a score between 0 and 1. **No Hugging Face token is needed.** Long inputs are truncated to 512 tokens.

The script:

1. Downloads the pretrained tokenizer and weights on the first run.
2. Loads the model and input onto the T4 GPU.
3. Runs inference and prints the label and score.
4. Saves the result as JSON in a persistent Modal Volume.

The first run takes longer while Modal builds the container and downloads the model. Later runs can reuse the cached files. Try another sentence:

```bash
modal run inference.py --text 'This movie was disappointing.'
```

Each run prints a ready-to-copy `modal volume get ...` command to download its result. The volume is named `myssp-demo-data`; model files live under `/models` and results under `/outputs` inside that volume. These files survive the container stopping. Teammates in the same workspace/environment use the same volume; each result gets a unique filename.

### Running again with Conda

Next time, open a terminal in this repository and run the following. You only need to create the environment and set the team token once.

```bash
conda activate myssp-gpu
modal token info
modal run gpu_check.py
modal run inference.py --text 'I really enjoyed this movie!'
```

Check the workspace shown by `modal token info` before running the GPU commands. If you chose venv, use its activation command from step 1 instead of `conda activate`.

## 5. Adapt it for your research

Open `inference.py` and follow the numbered comments:

- **Model:** change `MODEL_ID` for another compatible text classifier. For text generation, images, or audio, also replace the model class and input/output processing with the model's own example.
- **Packages:** add dependencies to `image.pip_install(...)`.
- **GPU:** start with `gpu="T4"`; change it only when your model needs more memory.
- **Files:** keep model caches and results under `/data`, the mounted Volume, and call `volume.commit()` after writing. Other container files are temporary.

For a private/gated model, obtain access first. Create a Modal Secret named `huggingface` in the dashboard with an `HF_TOKEN` key, then add `secrets=[modal.Secret.from_name("huggingface")]` to the existing `@app.function(...)` in `inference.py`. Keep the token value out of the code.

## Keep credentials private

- Do not commit the private quick-start PDF, token values, `.env` files, or `.modal.toml`. Do not include credentials in screenshots or notebooks.
- `.gitignore` excludes common secret files, the private guide's filename pattern, downloaded weights, and local results. It cannot detect credentials pasted into code or remove files already tracked by Git.
- Before committing or pushing, review `git status --short` and `git diff --cached`. If a credential was ever committed, revoke/rotate it and arrange removal from Git history; adding an ignore rule alone is insufficient.

## Budget and common fixes

The team guide allocates **US$100** of compute. Check Usage & Billing, or ask MRG for usage, and contact MRG before exceeding the allocation. The allocation alone does not guarantee an automatic spending cutoff. GPU, CPU, memory, and storage charges can apply; see [current pricing](https://modal.com/pricing).

| Problem | What to do |
| --- | --- |
| `modal` command not found | Activate your chosen environment (`conda activate myssp-gpu` or the venv command in step 1), then run `python -m pip install -r requirements.txt`. |
| Authentication fails or workspace is wrong | Run `modal token set` with the intended team's credentials, then `modal token info`. |
| First inference is slow | Allow the initial image build and model download to finish. |
| GPU is out of memory after changing the model | Use a smaller model or input, or select a GPU with more memory. |

Based on the private MRG quick-start guide; credentials are intentionally omitted. References: [Conda environments](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html), [token setup](https://modal.com/docs/cli/latest/token), [Volumes](https://modal.com/docs/guide/volumes), and [Secrets](https://modal.com/docs/guide/secrets).
