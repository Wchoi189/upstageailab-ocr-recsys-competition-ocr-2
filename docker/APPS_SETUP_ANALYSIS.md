# Apps Directory Setup Analysis

## ✅ Current Configuration (CORRECT)

Your Docker configuration is **already properly set up**:

### 1. Entrypoint Script ([docker/entrypoint.sh](docker/entrypoint.sh#L49))
```bash
FOLDERS=("archive" "data" "outputs" "packages")
```
- ✅ `apps` is **NOT** in the symlink array
- ✅ Comment at line 103 confirms: "apps is kept as a real directory for git tracking"

### 2. Docker Compose ([docker/docker-compose.yml](docker-compose.yml#L21-L25))
```yaml
volumes:
  # Project source code (includes apps/)
  - ..:/workspaces:cached

  # Anonymous volumes for node_modules (isolated from host)
  - /workspaces/node_modules
  - /workspaces/apps/ocr_inference_console/node_modules
  - /workspaces/apps/playground-console/node_modules
  - /workspaces/apps/agentqms-dashboard/frontend/node_modules
  - /workspaces/apps/mcp-visibility-extension/node_modules
```
- ✅ `apps/` source code synced from host (for git tracking)
- ✅ `node_modules` in dedicated Docker volumes (for performance)

### 3. Dockerignore ([.dockerignore](.dockerignore))
- ✅ `apps/` is **NOT excluded** - properly included in build context
- ✅ `node_modules` **IS excluded** - won't bloat build context

## 📁 How It Works

```
Host: /workspaces/apps/
├── __init__.py
├── agentqms-dashboard/
│   ├── frontend/
│   │   ├── src/               # ← Real files (git tracked)
│   │   └── node_modules/      # ← Docker volume (not on host)
├── ocr_inference_console/
│   ├── src/                   # ← Real files (git tracked)
│   └── node_modules/          # ← Docker volume (not on host)
└── ...

Container: /workspaces/apps/
├── Same structure
└── node_modules/ directories mounted from Docker volumes
```

## 🎯 What You're Already Achieving

1. **Source code in git**: `apps/` is a real directory, fully tracked
2. **Fast node_modules**: Isolated in Docker volumes (no host sync overhead)
3. **No symlinks**: `apps/` is NOT symlinked like `data/` or `outputs/`
4. **Clean builds**: Build context includes source but excludes node_modules

## ❓ Why You Might Think It's "Being Created"

If you see `apps/` appearing during build, it's because:

1. **During Build**: Dockerfile copies project → includes `apps/` source code ✅
2. **At Runtime**: Docker mounts volumes for `node_modules` ✅
3. **Both are correct** - you want source code in build, volumes at runtime

## 🔍 Verification Checklist

Run these commands to verify:

```bash
# 1. Confirm apps is a directory (not symlink)
test -L /workspaces/apps && echo "SYMLINK ❌" || echo "DIRECTORY ✅"

# 2. Check git tracking
git ls-files apps/ | head -5

# 3. Verify entrypoint.sh doesn't symlink it
grep 'FOLDERS=(' docker/entrypoint.sh

# 4. Check node_modules are volumes (NOT on host)
docker compose -f docker/docker-compose.yml config | grep -A 20 "volumes:"
```

## 🛠️ Only If You Have Issues

### Problem: apps/ is a symlink (shouldn't be)
```bash
# Remove symlink and restore from git
rm /workspaces/apps
git checkout apps/
```

### Problem: node_modules on host (should be in volumes)
```bash
# Remove from host - Docker will recreate in volumes
rm -rf /workspaces/apps/*/node_modules
docker compose -f docker/docker-compose.yml down
docker compose -f docker/docker-compose.yml up -d
```

### Problem: apps/ not in git
```bash
# Add to git
git add apps/
git commit -m "feat: Add apps/ source code to repository"
```

## 📝 Summary

**Your configuration is correct as-is!** The setup achieves exactly what you want:
- ✅ Source code stored inside project (git tracked)
- ✅ node_modules in dedicated volumes (performance)
- ✅ No symlinks for apps/ directory
- ✅ Clean separation of code and dependencies

**No changes needed** unless verification shows an actual problem.
