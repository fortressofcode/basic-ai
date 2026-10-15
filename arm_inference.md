**Complete steps to build and install llama.cpp (with MTP support) system-wide on NVIDIA DGX Spark (GB10 / aarch64, Ubuntu 24.04):**

### 1. Install dependencies
```bash
sudo apt update
sudo apt install -y git clang cmake libcurl4-openssl-dev libssl-dev build-essential
```

### 2. Clone the repository
```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp
```

### 3. Configure the build (CUDA + GB10 architecture)
```bash
cmake -B build \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA=ON \
  -DGGML_CURL=ON \
  -DGGML_RPC=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121a-real
```

### 4. Compile
```bash
cmake --build build --config Release -j$(nproc)
```

### 5. Install system-wide
```bash
sudo cmake --install build --config Release
sudo ldconfig
```

### 6. Verify
```bash
llama-server --version
```

### Optional: run with MTP
```bash
llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL \
  --host 0.0.0.0 \
  --port 30000 \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  -ngl 999
```

MTP is included in recent mainline builds (no extra compile flag needed). Enable it at runtime with an MTP-capable GGUF and the `--spec-type draft-mtp` flag.

### References:
https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF
https://unsloth.ai/docs/models/qwen3.8-next#mtp-guide
