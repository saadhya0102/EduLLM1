# Local training setup — RTX 4070 Laptop (8 GB)

Your machine: **RTX 4070 Laptop 8 GB** (Ada — bf16 + FlashAttention-2), Ryzen 9
8945HS, 31 GB RAM. This runs QLoRA locally: 3B is easy, **7B fits with Unsloth**
(tight, use the flags below).

**WSL2 is strongly recommended** over native Windows — the `unsloth` /
`bitsandbytes` / `triton` stack installs cleanly on Linux and is a constant
fight on native Windows. Your GPU passes through to WSL2 automatically.

---

## Option A — WSL2 (recommended)

### 1. Install WSL2 + Ubuntu (PowerShell, admin), once
```powershell
wsl --install -d Ubuntu
```
Reboot if prompted. Make sure your NVIDIA Windows driver is current (it is —
566.07) — the CUDA runtime is exposed to WSL2 through it; do **not** install a
separate NVIDIA driver inside Ubuntu.

### 2. In the Ubuntu shell, verify the GPU is visible
```bash
nvidia-smi          # should show the RTX 4070 Laptop, 8 GB
```

### 3. Python env + deps
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
cd /mnt/c/Users/hasto/EduLLM1          # the project, via the Windows mount
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
# PyTorch (CUDA 12.1 build) + training stack:
pip install "torch==2.4.*" --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install "trl<0.9.0" peft accelerate bitsandbytes datasets
# for the data-gen / eval steps (API calls):
pip install -r requirements-local.txt
```

### 4. Bring your API keys into the WSL2 shell (for probe / gen / eval)
```bash
export ANTHROPIC_AUTH_TOKEN='...'   # same values as on Windows
export ANTHROPIC_BASE_URL='...'
export OPENAI_API_KEY='...'
# (add these to ~/.bashrc to persist)
```

### 5. Train
```bash
python -m src.train --dev     # 3B smoke test (~5-10 min on current data)
python -m src.train           # 7B target (~15-25 min on current data)
```

> Working from `/mnt/c/...` is convenient but disk I/O is slower than the native
> WSL filesystem. For big datasets you can `cp -r` the project into `~/EduLLM1`
> instead; for this project size the mount is fine.

---

## Option B — native Windows (if you skip WSL2)
```powershell
cd C:\Users\hasto\EduLLM1
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
pip install "trl<0.9.0" peft accelerate bitsandbytes datasets
pip install -r requirements-local.txt
python -m src.train --dev
```
If `bitsandbytes` or `triton` errors out here, switch to WSL2 — it's not worth
debugging Windows wheels.

---

## 8 GB VRAM cheat-sheet
| Situation | Command |
|---|---|
| Smoke test (3B) | `python -m src.train --dev` |
| 7B target | `python -m src.train` |
| 7B won't fit (CUDA OOM) | `python -m src.train --max-seq-len 512 --grad-accum 16` |
| Still OOM | drop to `--dev` (3B), or close other GPU apps (browser, games) |
| Resume after interruption | add `--resume` |

Monitor VRAM in another shell with `watch -n1 nvidia-smi`. The script also prints
peak VRAM at the end.

## Model download
`src/train.py` pulls `unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit` (~5.5 GB) from
Hugging Face on first run and caches it. To pre-download into the project so the
first run is instant:
```bash
pip install huggingface_hub
huggingface-cli download unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit \
    --local-dir models/qwen2.5-math-7b-4bit
```
If that folder exists, `train.py` uses it automatically.
