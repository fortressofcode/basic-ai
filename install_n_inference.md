
**Environment:** Ubuntu 24.04 x86_64, working NVIDIA drivers, internet access, ~15–20 GB free disk, prefer source build for full CUDA control + community prebuilts as alternative, pure-text use of the exact quant you named, start at 8K context and scale toward 16K while staying under ~7 GB VRAM, Open WebUI as the chat frontend.

**References for all documentation below:**
- Official llama.cpp build docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md and https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md
- Unsloth Qwen3.5 docs + model card: https://unsloth.ai/docs/models/qwen3.5 and https://huggingface.co/unsloth/Qwen3.5-9B-GGUF
- Model file: Qwen3.5-9B-UD-Q4_K_XL.gguf ≈ 5.97 GB (Hugging Face / ModelScope listings)
- Community CUDA prebuilts: https://github.com/ai-dock/llama.cpp-cuda
- Open WebUI + llama.cpp integration: https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-llama-cpp and related docs
- VRAM / layer guidance: community analyses and llama.cpp multi-GPU / offload notes

### 1. Prerequisites (common to both methods)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Essential packages
sudo apt install -y git build-essential cmake curl libcurl4-openssl-dev pciutils wget

# Verify GPU
nvidia-smi
# Note compute capability (should be 8.9 for Ada Lovelace RTX 2000 Ada Laptop)
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
```

**Install NVIDIA CUDA Toolkit (only if needed — required for source build or if drivers alone are insufficient).**
First check whether a working toolkit is already present; if `nvcc` reports a recent version, you can skip this.
```bash
# Check for an existing CUDA toolkit (skip the install below if this prints a version)
nvcc --version 2>/dev/null || echo "CUDA toolkit not found — install it below"
```

If not present, install from NVIDIA's official Ubuntu 24.04 network repository (recommended — always pulls the latest compatible toolkit):
```bash
# 1. Add NVIDIA's CUDA repo keyring for Ubuntu 24.04 x86_64
cd ~
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# 2. Refresh package lists and install the toolkit
sudo apt update
sudo apt install -y cuda-toolkit
# (To pin a specific series matching your driver, use e.g. cuda-toolkit-12-8 instead.)

# 3. Add CUDA to your environment (append to ~/.bashrc to make it permanent)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 4. Verify
nvcc --version
```

Match the toolkit (12.x or 13.x series) to your installed driver — `nvidia-smi` shows the maximum CUDA version the driver supports. See the official guide: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04

For a hard ~7 GB VRAM ceiling on a laptop: monitor with `watch -n1 nvidia-smi` during runs, use `-ngl` (layers) to leave headroom, enable flash attention, and start with lower context. KV-cache quantization or unified memory can help further if needed.

### 2. Binary / Prebuilt Install (faster, community-provided)
Official llama.cpp GitHub releases provide Linux CPU/Vulkan binaries but **not** CUDA for Ubuntu. Use the actively maintained community CUDA builds that include compute capability 8.9:

```bash
# Example for a recent release (check https://github.com/ai-dock/llama.cpp-cuda/releases for the latest bXXXX and CUDA 12.8 amd64)
cd ~
wget https://github.com/ai-dock/llama.cpp-cuda/releases/download/bXXXX/llama.cpp-bXXXX-cuda-12.8-amd64.tar.gz   # replace with actual
tar -xzf llama.cpp-bXXXX-cuda-12.8-amd64.tar.gz
cd cuda-12.8   # or the extracted folder
# Binaries: llama-cli, llama-server, etc. are now ready
./llama-cli --version
```

Requirements noted in that repo: NVIDIA driver ≥ ~570 for CUDA 12.8, compute ≥ 7.5 (your 8.9 is covered). Place the binaries somewhere in PATH if desired.

Alternative package routes (may have limited or no CUDA):
- `brew install llama.cpp` (Homebrew on Linux)
- `conda install -c conda-forge llama.cpp` (or mamba/pixi)

### 3. Source Build (Recommended – full control, latest features, optimal for your GPU)
```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# CUDA build targeting your architecture (8.9) for faster compile + good performance
# Use native detection if preferred, or explicit architectures
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89" -DCMAKE_BUILD_TYPE=Release
# Optional: -DGGML_NATIVE=OFF for broader compatibility; add -DLLAMA_CURL=ON if desired

cmake --build build --config Release -j$(nproc)

# Binaries land in build/bin/
ls build/bin/llama-*
# Optional: copy to a permanent location or add to PATH
```

This matches the official CUDA section of the build.md documentation. Rebuild after `git pull` for updates.

### 4. Download the Model
```bash
# Install HF tools if needed
pip install -U huggingface_hub hf_transfer   # or use uv/pipx

# Download only the quant you specified (≈ 5.97 GB)
mkdir -p ~/models
cd ~/models
hf download unsloth/Qwen3.5-9B-GGUF \
  --include "Qwen3.5-9B-UD-Q4_K_XL.gguf" \
  --local-dir .
# Optional mmproj if you later want vision: --include "*mmproj-F16*"
```

(Direct wget/curl from the Hugging Face file URL also works.)

### 5. Run Commands (8K start → aim for 16K, stay near 7 GB)
Model weights ≈ 6 GB; remaining budget is for KV cache + overhead. Use full offload first (`-ngl 99` or `-ngl 999`), watch `nvidia-smi`, and drop layers or context if you approach 7 GB.

**Basic CLI test (8K):**
```bash
cd ~/llama.cpp   # or wherever your binaries are
./build/bin/llama-cli \
  -m ~/models/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  -c 8192 \
  -ngl 99 \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 \
  -p "Hello, introduce yourself briefly."
```

**Server mode (recommended for chat UI):**
```bash
./build/bin/llama-server \
  -m ~/models/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  -c 8192 \
  -ngl 99 \
  --host 127.0.0.1 --port 8080 \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 \
  --flash-attn   # helps with longer context / lower VRAM
  # Optional thinking (Qwen3.5-9B disables by default):
  # --chat-template-kwargs '{"enable_thinking":true}'
```

For 16K: change `-c 16384`. If VRAM exceeds ~7 GB:
- Reduce `-ngl` (e.g., start at 28–30 and climb while monitoring).
- Add `--cache-type-k q8_0 --cache-type-v q8_0` (or lower) for KV quant.
- Use environment variable `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` (trades some speed for system RAM spill).
- Lower concurrent slots (`-np 1`).

Unsloth-recommended sampling for non-thinking general use is close to the values above; adjust temperature/top-p for thinking mode as documented on their page.

Built-in web UI is available at http://127.0.0.1:8080 once the server is running.

### 6. Quick Open-Source Chat UI for Testing – Open WebUI
Lightweight, actively maintained, connects natively to llama-server’s OpenAI-compatible endpoint.

```bash
# Simple install (or use Docker for isolation)
pip install open-webui   # or: uv tool install open-webui / pipx
open-webui serve --host 127.0.0.1 --port 3000
```

Then:
1. Open http://127.0.0.1:3000 and create an admin account.
2. Admin Settings → Connections → OpenAI → Add:
   - URL: `http://127.0.0.1:8080/v1`
   - API Key: any string (e.g. `sk-no-key-required`)
   - Provider: llama.cpp (if listed)
3. Select the model and chat.

(Docker alternative is also well-documented on the Open WebUI site if you prefer containers.)

This combination (llama-server + Open WebUI) is the current best-practice quick test stack for local GGUF models.

### 7. Configure OpenCode to Use the Local Model
[OpenCode](https://opencode.ai) is a terminal-based AI coding agent. It can talk to any OpenAI-compatible endpoint, so point it at the running `llama-server` (`http://127.0.0.1:8080/v1`). Keep the server from §5 running.

```bash
# Install OpenCode (any one of these)
curl -fsSL https://opencode.ai/install | bash
# or: npm install -g opencode-ai
# or: brew install sst/tap/opencode
```

Add a custom local provider. Use `~/.config/opencode/opencode.json` for a global config, or an `opencode.json` in your project root for a per-project one:
```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llamacpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama.cpp (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "sk-no-key-required"
      },
      "models": {
        "Qwen3.5-9B-UD-Q4_K_XL": {
          "name": "Qwen3.5-9B (local)"
        }
      }
    }
  }
}
```

The model key must match what `llama-server` reports at `http://127.0.0.1:8080/v1/models` (by default the GGUF filename without extension). Then:
```bash
opencode        # launch the TUI
# Use /models (or the model picker) and select "llama.cpp (local) → Qwen3.5-9B (local)"
```

Notes:
- `llama-server` ignores the API key, but OpenCode expects the field present — any non-empty string works.
- For agentic/tool-use workflows, start the server with a bigger context (`-c 16384`) since coding prompts consume tokens quickly; watch VRAM per §5.
- The `npm` package (`@ai-sdk/openai-compatible`) is fetched automatically by OpenCode on first use.

### Next Steps / Troubleshooting
- After first successful 8K run, experiment with 12K → 16K while watching VRAM.
- If OOM occurs, drop layers first, then context, then try KV quant.
- For laptop power/thermals: use `nvidia-smi -pl <watts>` or the system’s power profile.
- Update: `git pull` + rebuild, or re-download newer community binaries.
