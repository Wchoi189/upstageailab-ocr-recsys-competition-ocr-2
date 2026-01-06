# 🔧 Cloud Development Setup Fixes

## Current Status: ❌ WILL FAIL

Your `.devcontainer/devcontainer.json` has GPU settings and local mounts that won't work in GitHub Codespaces.

## Quick Fix (5 minutes)

### Step 1: Backup Current Config
```bash
mv .devcontainer/devcontainer.json .devcontainer/devcontainer.local.json
```

### Step 2: Use Cloud-Optimized Config
```bash
mv .devcontainer/devcontainer.cloud.json .devcontainer/devcontainer.json
```

### Step 3: Verify & Commit
```bash
# Check the config
cat .devcontainer/devcontainer.json

# Commit the change
git add .devcontainer/
git commit -m "fix(devcontainer): use cloud-optimized config for Codespaces compatibility"
git push
```

### Step 4: Make GHCR Image Public
**CRITICAL**: Your image must be public for Codespaces to access it.

1. Go to: https://github.com/users/Wchoi189/packages/container/upstageailab-ocr-recsys-competition-ocr-2/settings
2. Scroll to "Danger Zone"
3. Click "Change visibility" → "Public"
4. Confirm

**Alternative**: Add build fallback (slower but works without public image):
- The cloud config already has a build fallback
- First start will take 10-15 minutes to build
- Subsequent starts use the cached image

---

## What Was Fixed

### ✅ Removed GPU Requirements
```diff
- "runArgs": ["--gpus", "all", "--shm-size=8gb"]
+ // NO runArgs - cloud compatible
```

### ✅ Removed Local Mounts
```diff
- "mounts": ["source=${localWorkspaceFolder}/data..."]
+ // NO mounts - cloud environments don't have local filesystem
```

### ✅ Added Build Fallback
```json
"build": {
    "dockerfile": "../docker/Dockerfile",
    "context": "..",
    "target": "development"
}
```

### ✅ Optimized Lifecycle Commands
- Prevents 10-minute timeout
- Faster dependency installation
- Graceful failure handling

### ✅ Added Claude Dev Extension
For MCP server support in Codespaces

---

## Testing Your Setup

### Test Locally (Docker Desktop)
```bash
# Use the local config
code . --folder-uri vscode-remote://dev-container+${PWD}/.devcontainer/devcontainer.local.json
```

### Test in Codespaces
1. Push changes to GitHub
2. Go to repo → Code → Codespaces
3. Create codespace
4. Wait for setup (first time: ~5-10 min)
5. Check output for errors

### Verify Success
```bash
# In the container, run:
python scripts/validate_environment.py
uv pip list | head -20
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check MCP servers
bash .devcontainer/scripts/start-mcp-servers.sh
```

---

## Additional Considerations for Cloud Development

### 1. Data Access
**Problem**: No local `./data` directory in cloud

**Solutions**:
- **Option A**: Commit small sample data to repo
  ```bash
  # Add sample data (< 100MB)
  git add data/samples/
  ```

- **Option B**: Download in postCreateCommand
  ```json
  "postCreateCommand": "wget https://example.com/data.zip && unzip data.zip"
  ```

- **Option C**: Use cloud storage (S3/GCS)
  ```bash
  # Add credentials as Codespaces secrets
  # Download data on startup
  ```

### 2. GPU Workloads
**Problem**: Codespaces has no GPU

**Solutions**:
- Use CPU-only for development/testing
- Use smaller models for validation
- Run training jobs separately (Colab, SageMaker, etc.)
- Add fallback code:
  ```python
  device = "cuda" if torch.cuda.is_available() else "cpu"
  ```

### 3. Large Dependencies
**Problem**: PyTorch + ML packages = slow install

**Solutions**:
- ✅ Use pre-built image (already configured)
- Enable dependency caching in image builds
- Use `--no-deps` for dev installs
- Pin versions in pyproject.toml

### 4. Codespaces Storage Limits
**Free Tier**: 15GB storage

**Monitor usage**:
```bash
df -h /workspaces
du -sh ~/.cache/uv
```

**Clean up**:
```bash
# Clear UV cache
uv cache clean

# Remove old venvs
rm -rf .venv
```

### 5. Port Forwarding
Already configured for:
- 8000: API Server
- 8501: Streamlit
- 6006: TensorBoard

Access via: `https://<codespace>-8000.app.github.dev`

---

## Troubleshooting

### Container won't start
**Check**: Is image public?
```bash
docker pull ghcr.io/wchoi189/upstageailab-ocr-recsys-competition-ocr-2:latest
```

**Fix**: Make image public OR use build fallback

### Dependencies fail to install
**Check**: Timeout (10 min limit)?

**Fix**: Use pre-built image with dependencies OR optimize postCreateCommand

### Python extension not working
**Check**: Is .venv created?
```bash
ls -la .venv/bin/python
```

**Fix**: Run `uv sync` manually

### MCP servers not found
**Check**: Are they committed?
```bash
ls -la */mcp_server.py
```

**Fix**: Commit MCP server files to repo

---

## Cost Optimization

### Free Tier Limits
- 120 core-hours/month
- 4-core machine = 30 hours/month free
- 15GB storage (included)

### Tips
1. Stop Codespaces when not in use
2. Set auto-suspend to 30 minutes
3. Use smaller machine for non-ML work
4. Delete unused Codespaces
5. Use pre-built images (faster = cheaper)

---

## Summary

### Before (Will Fail ❌)
- GPU requirements ❌
- Local filesystem mounts ❌
- May timeout on setup ❌
- Image might be private ❌

### After (Will Work ✅)
- Cloud-compatible ✅
- No GPU/mount dependencies ✅
- Optimized setup commands ✅
- Build fallback included ✅
- MCP servers ready ✅

**Next Steps**:
1. Apply fixes above
2. Make image public
3. Test in Codespaces
4. Report any issues

**Questions?** Check:
- `.devcontainer/README.md`
- `.devcontainer/CODESPACES_SETUP.md`
- [Devcontainer Docs](https://containers.dev/)
