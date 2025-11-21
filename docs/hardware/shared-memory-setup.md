# Shared Memory Setup for WSL/Docker

## What is Shared Memory?

**Shared memory (`/dev/shm`) is RAM-based, NOT GPU VRAM.**

- It's a temporary filesystem that uses your system RAM
- PyTorch DataLoader workers use it to share data between processes
- Default size in WSL/Docker: 64MB (very limited)
- With 60GB RAM, you can safely increase it to 2-4GB

## Why Increase Shared Memory?

With limited shared memory (64MB), you'll get bus errors when using multiple DataLoader workers:
```
ERROR: Unexpected bus error encountered in worker.
This might be caused by insufficient shared memory (shm).
```

## Quick Fix (Temporary)

Increase shared memory temporarily (until WSL restart):

```bash
# For 60GB RAM system, use 4-8GB
sudo mount -o remount,size=4G /dev/shm
df -h /dev/shm  # Verify it's now 4GB
```

## Permanent Fix (WSL)

Make the change permanent by editing `/etc/wsl.conf`:

```bash
# Edit the file
sudo nano /etc/wsl.conf

# Add this content (4GB for 60GB RAM system):
[boot]
command = "mount -o remount,size=4G /dev/shm"

# Or use 8GB for maximum headroom:
# command = "mount -o remount,size=8G /dev/shm"

# Save and exit (Ctrl+X, Y, Enter)
```

Then restart WSL from Windows PowerShell:
```powershell
wsl --shutdown
# Then restart your WSL terminal
```

## Recommended Sizes

| RAM Available | Recommended Shared Memory | Max Workers |
|---------------|---------------------------|-------------|
| 8GB | 512MB | 4-6 workers |
| 16GB | 1GB | 8-10 workers |
| 32GB | 2-4GB | 12-16 workers |
| 60GB+ | **4-8GB** | 12-16 workers |

With 60GB RAM, you can safely use **4-8GB** for shared memory.

### Optimal Size Calculation

For OCR training with:
- 12 workers
- `prefetch_factor: 3`
- `batch_size: 4-6`
- `persistent_workers: true`

**Memory Requirements:**
- Per image (640×640×3, FP32): ~5MB
- Per batch (4 images): ~20MB
- Per worker (3 prefetch batches): ~60MB
- Total for 12 workers: ~720MB
- Plus overhead: ~500MB
- **Total needed: ~1.2-1.5GB**

**Recommended with safety margin:**
- **Minimum: 2GB** (works but tight)
- **Optimal: 4-8GB** (comfortable headroom, allows scaling)
- **Maximum: 8-16GB** (if you want to experiment with larger batches/workers)

For your 60GB RAM system, **4-8GB is optimal** for OCR training.

## Verify Setup

After increasing shared memory:

```bash
# Check current size
df -h /dev/shm

# Should show something like:
# Filesystem      Size  Used Avail Use% Mounted on
# shm             4.0G     0  4.0G   0% /dev/shm
```

## Using the Optimized Configurations

After increasing shared memory, you can use the full RTX 3060 configuration:

```bash
# Use the optimized config with 12 workers
python runners/train.py dataloaders=rtx3060_16core batch_size=4
```

This will use:
- 12 workers for training/validation
- `persistent_workers: true` for faster epoch transitions
- `prefetch_factor: 3` for better GPU utilization

## Docker Container Configuration

**Important:** If you're using Docker containers (like VS Code Dev Containers), `/etc/wsl.conf` won't work because containers have their own shared memory namespace.

### Configure Shared Memory in Docker

Add `shm_size` to your `docker-compose.yml`:

```yaml
services:
  dev:
    # ... other config ...
    shm_size: '4gb'  # Set shared memory size for DataLoader workers
```

Or use Docker run command:
```bash
docker run --shm-size=4g ...
```

**After updating docker-compose.yml:**
1. Rebuild and restart the container:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

2. Verify shared memory size:
   ```bash
   df -h /dev/shm
   # Should show 4.0G
   ```

### Get Root Access in Containers

The `vscode` user already has passwordless sudo configured in the Dockerfile. To use it:

```bash
# You should be able to use sudo without password
sudo mount -o remount,size=4G /dev/shm

# If that doesn't work, check sudoers:
sudo cat /etc/sudoers | grep vscode
# Should show: vscode ALL=(ALL) NOPASSWD: ALL
```

**If sudo still requires a password:**
1. Check if you're in the container:
   ```bash
   test -f /.dockerenv && echo "In Docker" || echo "Not in Docker"
   ```

2. Switch to root user (if needed):
   ```bash
   # In Docker, you might need to exec as root
   docker exec -u root -it <container_name> bash

   # Or if you have root access in the container:
   su - root
   ```

3. Remount shared memory:
   ```bash
   mount -o remount,size=4G /dev/shm
   ```

## Troubleshooting

### VS Code Terminal Shows Different Size

If your regular WSL terminal shows 8GB but VS Code's integrated terminal still shows 64MB:

**This is normal!** VS Code's terminal uses a different WSL session/namespace. Solutions:

1. **Best: Use permanent fix** (applies to all sessions):
   - In your regular WSL terminal (where mount works), create `/etc/wsl.conf`:
     ```bash
     sudo nano /etc/wsl.conf
     ```
   - Add this content:
     ```ini
     [boot]
     command = "mount -o remount,size=4G /dev/shm"
     ```
   - Save and exit (Ctrl+X, Y, Enter)
   - Restart WSL from Windows PowerShell:
     ```powershell
     wsl --shutdown
     ```
   - Reopen VS Code — all terminals will show 4GB

2. **Or restart VS Code's WSL connection:**
   - Close VS Code completely
   - Restart WSL: `wsl --shutdown` (from Windows PowerShell)
   - Reopen VS Code (this will pick up the mount from your regular terminal)

3. **Workaround: Use low_shm config** (if permanent fix doesn't work):
   - Use the `low_shm` configuration that works with 64MB:
     ```bash
     python runners/train.py dataloaders=low_shm batch_size=4
     ```
   - This uses fewer workers but avoids bus errors

### Still Getting Bus Errors?

1. **Check shared memory size:**
   ```bash
   df -h /dev/shm
   ```

2. **If still 64MB, the remount didn't work:**
   - Make sure you used `sudo`
   - Check if WSL restarted (temporary changes are lost on restart)
   - Use the permanent fix above
   - If using VS Code, re-run the mount command in VS Code's terminal

3. **If errors persist with 4GB shm:**
   - Reduce `num_workers` to 8-10
   - Reduce `prefetch_factor` to 2
   - Or increase shared memory to 8GB

### Check Current Usage

Monitor shared memory usage during training:

```bash
# Watch shared memory usage
watch -n 1 'df -h /dev/shm'
```

If usage approaches 100%, increase the size or reduce workers.
