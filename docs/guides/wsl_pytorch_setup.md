# WSL GPU Setup & PyTorch Version Playbook

Use this guide to keep a Windows 11 → WSL 2 → Docker environment healthy when running GPU workloads with PyTorch, and to switch PyTorch versions safely.

---

## 1. Keep Host & WSL Up to Date

1. **Update the NVIDIA driver on Windows**  
   Install the latest Studio or Game Ready driver (≥ 552.xx) for the RTX 3060, then reboot.

2. **Refresh WSL kernel & GPU forwarding**
   ```powershell
   wsl --update
   wsl --shutdown
   ```
   Relaunch WSL after the shutdown.

3. **Verify GPU visibility inside WSL**
   ```bash
   uname -r
   nvidia-smi
   sudo apt update && sudo apt upgrade -y
   ```

---

## 2. Prepare Docker for GPU Workloads

1. **Install / update NVIDIA Container Toolkit**
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/nvidia-container-toolkit.list |         sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt update
   sudo apt install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

2. **Sanity check GPU passthrough**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

---

## 3. Manage PyTorch Versions inside WSL

### 3.1 Optional: create a clean environment
```bash
python3 -m venv ~/.venvs/pytorch-env
source ~/.venvs/pytorch-env/bin/activate
pip install --upgrade pip
```

### 3.2 Remove any existing PyTorch build
```bash
pip uninstall -y torch torchvision torchaudio
```

### 3.3 Install the desired build

| Scenario | Command |
|----------|---------|
| Recommended Ampere (CUDA 12.1) | `pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121` |
| Legacy CUDA 11.8 | `pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118` |
| CPU-only | `pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu` |
| Nightly (unstable) | `pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121` |

> Update `pyproject.toml` / `requirements.txt` if you want builds to stay consistent.

### 3.4 Confirm everything works
```bash
python - <<'PYTORCH_CHECK'
import torch, torchvision
print("Torch      :", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA avail :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device     :", torch.cuda.get_device_name(0))
PYTORCH_CHECK
```

*Debug tip:* set `export TORCH_USE_CUDA_DSA=1` before training to enable device-side assertions.

---

## 4. Rebuild Project Images (if applicable)

1. Adjust Dockerfiles to a CUDA 12.1/12.4 base image, e.g. `FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04`.
2. Rebuild after updating dependency pins:
   ```bash
   docker build -t your-image .
   ```
3. Launch with GPU access:
   ```bash
   docker run --rm --gpus all -it your-image bash
   ```

---

## 5. Quick Validation Before Full Training
```bash
python runners/train.py --trainer.fast_dev_run true
```

Lightning's fast dev run compiles the model and processes a single batch to catch issues early.

---

## 6. Troubleshooting Notes

- `CUDNN_STATUS_EXECUTION_FAILED` usually points to driver/CUDA mismatches or corrupted tensors.
- Ensure the CUDA version reported by `nvidia-smi` matches the PyTorch wheels you install.
- Remove GPU overclocks while debugging stability problems.
- If failures persist, keep device assertions enabled and inspect intermediate tensors for NaN/Inf values.

---

Forward this document to anyone maintaining the environment to keep WSL GPU setups stable and PyTorch versions under control.
