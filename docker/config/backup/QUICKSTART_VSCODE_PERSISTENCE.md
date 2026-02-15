# Quick Start: /home/vscode Persistence

## What Changed

✅ Added persistent volume for `/home/vscode`
✅ Antigravity IDE conversations, plans, and debugging info now survive container resets
✅ Shell history, caches, and configs are preserved

## Apply Changes

### From Windows Host

```powershell
# Stop container
wsl bash -c "cd /data/upstageailab-ocr-recsys-competition-ocr-2/docker && docker compose down"

# Start with new volume (will be created automatically)
wsl bash -c "cd /data/upstageailab-ocr-recsys-competition-ocr-2/docker && docker compose up -d"
```

### From Inside Container/WSL

```bash
cd /workspaces/docker
docker compose down
docker compose up -d
```

## Verify

Check if volume was created:
```bash
docker volume ls | grep vscode_home
```

Check volume status:
```bash
./docker/manage-vscode-home.sh info
```

## That's It!

Going forward, your IDE state will **automatically persist** across:
- Container restarts
- Container rebuilds
- System reboots

## Optional: Backup Current State First

If you want to preserve your **current** Antigravity conversations before applying:

```bash
# Create one-time backup
docker exec ocr-recsys-dev tar czf /tmp/vscode_backup.tar.gz -C /home vscode

# Copy to project
docker cp ocr-recsys-dev:/tmp/vscode_backup.tar.gz ./docker/backups/

# Then apply changes normally
cd docker && docker compose down && docker compose up -d

# Restore if needed (see VSCODE_HOME_PERSISTENCE.md)
```

## Files Changed

1. ✅ [docker/docker-compose.yml](docker-compose.yml) - Added `vscode_home` volume
2. ✅ [docker/manage-vscode-home.sh](manage-vscode-home.sh) - Management script
3. ✅ [docker/VSCODE_HOME_PERSISTENCE.md](VSCODE_HOME_PERSISTENCE.md) - Full documentation

## Next Steps

**Just restart your container!** Everything will work automatically.

For detailed information, see: [VSCODE_HOME_PERSISTENCE.md](VSCODE_HOME_PERSISTENCE.md)
