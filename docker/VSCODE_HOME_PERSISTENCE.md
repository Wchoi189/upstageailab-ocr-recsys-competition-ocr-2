# /home/vscode Persistence Configuration

## Overview

The `/home/vscode` directory is now **persisted across container resets** using a Docker named volume. This preserves:

- ✅ **VS Code Server state** (`.vscode-server/`)
- ✅ **Antigravity IDE conversations, plans, debugging info** (stored in VS Code data)
- ✅ **Copilot chat history** and context
- ✅ **Shell history** (`.bash_history`)
- ✅ **Cached dependencies** (`.cache/`, `.npm/`, `.pyenv/`)
- ✅ **User configurations** (`.config/`, `.local/`)
- ✅ **Python environments** managed by pyenv/UV
- ✅ **pnpm global packages** and cache

## Configuration

### Docker Compose ([docker-compose.yml](docker-compose.yml))

```yaml
volumes:
  - vscode_home:/home/vscode  # Named volume persists across container resets

volumes:
  vscode_home:  # Defined at bottom of file
```

### What's Excluded

The following are **NOT persisted** (by design):
- `.ssh-mount/` - Mounted from host `~/.ssh/` (refreshed each start)
- Large build caches - Auto-cleaned on backup

## Usage

### Normal Operation

**No action needed!** The volume is automatically created and persisted when you:
```bash
cd docker
docker compose up -d
```

Your IDE state will survive:
- Container restarts (`docker compose restart`)
- Container rebuilds (`docker compose up -d --build`)
- Docker daemon restarts

### Management Commands

Use the helper script: `./docker/manage-vscode-home.sh`

#### Check Status
```bash
./docker/manage-vscode-home.sh info
```
Shows volume size, key directories, and usage stats.

#### Create Backup
```bash
./docker/manage-vscode-home.sh backup
```
Creates a timestamped backup in `docker/backups/`:
- `vscode_home_20260208_143022.tar.gz`
- Excludes large caches and mount points
- Useful before major changes or migrations

#### Restore Backup
```bash
./docker/manage-vscode-home.sh restore docker/backups/vscode_home_20260208_143022.tar.gz
```
Restores a previous backup (with confirmation prompt).

#### Reset Volume
```bash
./docker/manage-vscode-home.sh reset
```
**⚠️ WARNING:** Deletes entire volume (requires typing "DELETE" to confirm).

Use when:
- Testing fresh environment setup
- Troubleshooting persistent corruption
- Migrating to new volume

#### Clone to Host
```bash
./docker/manage-vscode-home.sh clone
```
Creates a one-time copy in `docker/vscode_home_clone_*/` for inspection.

## Migration Guide

### From Non-Persisted to Persisted

If you're upgrading from a setup without persistence:

1. **Start container normally** (volume will be created empty):
   ```bash
   cd docker
   docker compose up -d
   ```

2. **Your IDE will start fresh** - This is expected on first run

3. **Going forward** - All state will be preserved automatically

### Optional: Preserve Existing State

If you want to keep current state before applying changes:

```bash
# 1. Create backup of current state BEFORE volume changes
docker exec ocr-recsys-dev tar czf - -C /home vscode > /tmp/vscode_pre_persistence.tar.gz

# 2. Apply docker-compose.yml changes (already done)

# 3. Restart container (creates new volume)
cd docker
docker compose down
docker compose up -d

# 4. Restore backup into new volume
docker exec -i ocr-recsys-dev tar xzf - -C /home < /tmp/vscode_pre_persistence.tar.gz
docker exec ocr-recsys-dev sudo chown -R vscode:vscode /home/vscode

# 5. Restart to apply
docker compose restart
```

## Troubleshooting

### Volume is growing too large

Check what's consuming space:
```bash
./docker/manage-vscode-home.sh info
```

Manually clean caches in container:
```bash
docker exec ocr-recsys-dev bash -c '
  rm -rf ~/.cache/uv/builds-*
  rm -rf ~/.cache/pip/http
  rm -rf ~/.npm/_cacache
'
```

### Need to start completely fresh

```bash
./docker/manage-vscode-home.sh reset
```

### Want to migrate to different host

1. Create backup:
   ```bash
   ./docker/manage-vscode-home.sh backup
   ```

2. Copy backup file to new host

3. On new host, restore:
   ```bash
   ./docker/manage-vscode-home.sh restore path/to/backup.tar.gz
   ```

### Permissions issues

Fix ownership:
```bash
docker exec ocr-recsys-dev sudo chown -R vscode:vscode /home/vscode
```

## Technical Details

### Volume Location

Docker stores the volume at:
- **Linux**: `/var/lib/docker/volumes/ocr-dev_vscode_home/_data`
- **Windows/Mac**: Inside Docker Desktop VM

### Performance Impact

- **Read/Write**: Native Docker volume performance (fast)
- **Size**: Grows with usage (~500MB-2GB typical)
- **Backup**: Compressed backups are 100-500MB

### Interaction with Entrypoint

The entrypoint script ([entrypoint.sh](entrypoint.sh)) still:
- Copies SSH keys from `.ssh-mount` to `.ssh` (refreshed each start)
- Installs global tools
- Sets up symlinks for project data

These operations work correctly with the persisted home directory.

## Benefits

✅ **No more lost conversations** - Antigravity IDE state persists
✅ **Faster startups** - Dependencies and caches preserved
✅ **Consistent environment** - Shell history and configs retained
✅ **Easy backups** - Built-in backup/restore tools
✅ **Safe resets** - Clean slate when needed without losing project data

## See Also

- [Docker Compose Configuration](docker-compose.yml)
- [Entrypoint Script](entrypoint.sh)
- [Management Script](manage-vscode-home.sh)
