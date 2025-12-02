# WSL GPU Setup & PyTorch Version Playbook (Agent Script)

This document is an executable checklist for the AI agent operating inside the Windows 11 → WSL 2 → Docker environment. Follow the tasks in order and update the progress tracker as you complete each step.

---

## Progress Tracker

- [ ] **Task 1** – Verify host drivers and WSL kernel
- [ ] **Task 2** – Update NVIDIA container runtime
- [ ] **Task 3** – Prepare a clean Python environment
- [ ] **Task 4** – Install GPU-compatible PyTorch ≥ 2.6.0
- [ ] **Task 5** – Rebuild/verify project Docker images (if used)
- [ ] **Task 6** – Run fast validation and capture diagnostics

> Mark each item `[x]` once the associated commands complete without errors.

---

## Task 1 – Host & WSL Health Check

1. Update Windows NVIDIA driver (Studio/Game Ready ≥ 552.xx) and reboot the host.
2. Inside PowerShell (host):
   ```powershell
   wsl --update
   wsl --shutdown
   ```
3. Relaunch WSL and confirm GPU visibility:
   ```bash
   uname -r
   nvidia-smi
   sudo apt update && sudo apt upgrade -y
   ```
4. If `nvidia-smi` fails, reinstall the Windows driver before proceeding.

---

## Task 2 – NVIDIA Container Toolkit

1. Install/refresh the toolkit (run inside WSL):
   ```bash
   distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey |         sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/nvidia-container-toolkit.list |         sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt update
   sudo apt install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
2. Validate GPU passthrough:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```
   > Proceed only if the container sees the GPU.

---

## Task 3 – Python Environment Prep

1. (Optional) create a dedicated virtual environment:
   ```bash
   python3 -m venv ~/.venvs/pytorch-env
   source ~/.venvs/pytorch-env/bin/activate
   pip install --upgrade pip
   ```
2. Remove any pre-existing PyTorch packages:
   ```bash
   pip uninstall -y torch torchvision torchaudio || true
   ```

---

## Task 4 – Install PyTorch ≥ 2.6.0 (GPU Compatible)

| Scenario | Command |
|----------|---------|
| **Ampere Recommended (CUDA 12.4)** | `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124` |
| CUDA 12.1 fallback | `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121` |
| CPU-only build | `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu` |
| Nightly (unstable) | `pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124` |

> Ensure the chosen CUDA wheel matches the version reported by `nvidia-smi`. For the RTX 3060, the CUDA 12.4 wheel is preferred when available.

### Verification
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
If CUDA is not available, stop and investigate driver/container setup.

*Debug option:* `export TORCH_USE_CUDA_DSA=1` before running training to enable device-side assertions.

---

## Task 5 – Project Image Rebuild (Skip if not using Docker)

1. Update Dockerfiles to a matching CUDA base image, e.g.:
   ```dockerfile
   FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
   ```
2. Rebuild the project image:
   ```bash
   docker build -t ocr-wsl-gpu .
   ```
3. Smoke test GPU access in the new image:
   ```bash
   docker run --rm --gpus all -it ocr-wsl-gpu nvidia-smi
   ```

---

## Task 6 – Fast Validation & Diagnostics

1. Run Lightning’s fast dev run to ensure training compiles:
   ```bash
   python runners/train.py --trainer.fast_dev_run true
   ```
2. If the run fails, capture logs, CUDA driver info, and PyTorch version:
   ```bash
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())"
   nvidia-smi
   ```
3. Update the progress tracker and log any issues in the bug report template (`docs/bug_reports/`).

---

### Notes & Reminders

- Keep backups of `~/.bashrc`, `~/.gitconfig`, `~/.ssh`, `.env` files, and other local settings before destroying containers.
- Remove GPU overclock settings while diagnosing stability issues.
- If you need to rerun a task, reset its checkbox in the tracker and repeat the commands.

Once all tasks are checked, the environment is ready for full training runs with a secure PyTorch build (≥ 2.6.0).
