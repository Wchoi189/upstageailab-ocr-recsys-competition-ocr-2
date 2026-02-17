#!/bin/bash
set -e

# Container Paths Overview:
# - /workspaces → Project root (/data/upstageailab-ocr-recsys-competition-ocr-2)
# - /parent → Data drive root (/data) - provides access to sibling projects
# - /mnt/external_artifacts → External storage (/data/project-artifacts/ocr-external-storage)

# 1. Ensure the user has a password (vital for SSH)
echo "Setting vscode password..."
echo "vscode:vscode" | sudo chpasswd

# 2. Key Import Logic (Run on every start to catch new keys)
SSH_DIR="/home/vscode/.ssh"
MOUNT_DIR="/home/vscode/.ssh-mount"

# Ensure SSH dir exists and has correct ownership
if [ ! -d "$SSH_DIR" ]; then
    mkdir -p "$SSH_DIR"
    sudo chown vscode:vscode "$SSH_DIR"
    chmod 700 "$SSH_DIR"
fi

# Ensure authorized_keys exists
touch "$SSH_DIR/authorized_keys"
chmod 600 "$SSH_DIR/authorized_keys"
sudo chown vscode:vscode "$SSH_DIR/authorized_keys"

if [ -d "$MOUNT_DIR" ]; then
    echo "Importing public keys from host..."
    # Loop through public keys only
    for key in "$MOUNT_DIR"/*.pub; do
        if [ -f "$key" ]; then
            cat "$key" | tr -d '\r' >> "$SSH_DIR/authorized_keys"
            # Ensure a newline exists
            echo "" >> "$SSH_DIR/authorized_keys"
        fi
    done
fi

# 3. ROBUST SYMLINKING
# Source: Where the heavy data lives (The Bridge)
# Note: SOURCE_ROOT is mounted from host /data/project-artifacts/ocr-external-storage
# /parent directory gives access to entire /data drive (/parent = /data on host)
# Destination: Your workspace
SOURCE_ROOT="/mnt/external_artifacts"
DEST_ROOT="/workspaces"
# Note: "apps" removed from FOLDERS - keep source code in project for git tracking
FOLDERS=("archive" "data" "outputs" "packages")

if [ -d "$SOURCE_ROOT" ]; then
    echo "🔗 Linking external artifacts..."

    # --- MIGRATION LOGIC (Run once if needed) ---
    # Move root-level artifacts to nested locations if they exist in the old place

    # 1. Logs & WandB -> outputs/
    for item in "lightning_logs" "wandb" "hydra_outputs"; do
        if [ -d "$SOURCE_ROOT/$item" ]; then
            echo "  🚚 Migrating $item to outputs/$item..."
            mkdir -p "$SOURCE_ROOT/outputs"
            # Move content if dest doesn't exist, or merge? safer to move if dest absent
            if [ ! -d "$SOURCE_ROOT/outputs/$item" ]; then
                mv "$SOURCE_ROOT/$item" "$SOURCE_ROOT/outputs/$item"
            else
                echo "    ⚠️ Destination $SOURCE_ROOT/outputs/$item already exists. Check manually."
            fi
        fi
    done


    # --- END MIGRATION ---

    for folder in "${FOLDERS[@]}"; do
        SRC="$SOURCE_ROOT/$folder"
        DEST="$DEST_ROOT/$folder"

        if [ -d "$SRC" ]; then
            # If the destination exists (folder or broken link), move it aside
            # Prepare destination
            if [ -L "$DEST" ]; then
                # Only remove if it's a symlink
                rm "$DEST"
            elif [ -d "$DEST" ]; then
                # Backup real directories
                echo "  📦 Backing up existing directory: $folder"
                mv "$DEST" "${DEST}_backup_$(date +%s)"
            elif [ -e "$DEST" ]; then
                # Remove files
                rm "$DEST"
            fi

            # Create the fresh, clean symlink
            ln -s "$SRC" "$DEST"
            echo "  ✅ Linked: $folder -> $SRC"
        else
            echo "  ⏭️  Skipping $folder (source directory not found)"
        fi
    done

    # Cleanup old symlinks that are no longer in FOLDERS (extensions, hydra_outputs, etc)
    # We do this carefully. Note: apps is kept as a real directory for git tracking
    for old in "extensions" "hydra_outputs" "lightning_logs" "wandb"; do
        if [ -L "$DEST_ROOT/$old" ]; then
             echo "  🧹 Cleaning up deprecated symlink: $old"
             rm "$DEST_ROOT/$old"
        fi
    done

    # Add project bin directories to PATH for services
    # Note: /parent maps to /data on host, giving access to sibling projects
    export PATH="/home/vscode/.local/share/pnpm:/workspaces/bin:/workspaces/AgentQMS/bin:/parent/repomix/bin:$PATH"

    # Nested Symlinks for convenience (optional, but requested by user config implied?)
    # paths.yaml now uses ${output_dir}/wandb, so /workspaces/outputs/wandb is valid.
    # No extra symlinks needed if the migration worked and 'outputs' is linked.

    echo "🔗 Symlink setup complete!"
else
    echo "⚠️  External artifacts directory not found at $SOURCE_ROOT"
    echo "   Skipping symlink creation."
fi

# 3.5. Install Global Tools from /parent (if available)
PARENT_DIR="/parent"
if [ -d "$PARENT_DIR" ]; then
    echo "🛠️  Checking for local tools in $PARENT_DIR..."

    # Repomix
    if [ -d "$PARENT_DIR/repomix" ]; then
        echo "  📦 Installing repomix from local source..."
        # Install as vscode user, skip global link (bin is already in PATH)
        # We use explicit install/build steps and swallow errors solely for preventing crash
        sudo -u vscode bash -c "cd '$PARENT_DIR/repomix' && pnpm install && pnpm run build" || echo "  ❌ Failed to install repomix (non-fatal)"
    else
        # Fallback: install repomix globally if local source not available
        echo "  📦 Installing repomix globally..."
        sudo -u vscode bash -c 'export PATH="/home/vscode/.local/share/pnpm:$PATH" && pnpm add -g repomix' || echo "  ❌ Failed to install repomix globally (non-fatal)"
    fi

    # Spec-kit (requires Python 3.11+, use project venv)
    if [ -d "$PARENT_DIR/spec-kit" ] && [ -f "/workspaces/.venv/bin/python" ]; then
        echo "  📦 Installing spec-kit in project venv..."
        uv pip install -e "$PARENT_DIR/spec-kit" --python /workspaces/.venv/bin/python || echo "  ❌ Failed to install spec-kit (non-fatal)"
    fi

    # Qwen Code CLI (API Version)
    # Ensure qwen-code is installed/updated globally for the user
    echo "  📦 Checking/Updating qwen-code CLI..."

    # Ensure local bin dir exists
    sudo -u vscode mkdir -p /home/vscode/.local/share/pnpm

    # Configure pnpm (ignore if this fails)
    sudo -u vscode pnpm config set global-bin-dir /home/vscode/.local/share/pnpm || true

    # Install with explicit PATH, allowing failure. Using @latest to ensure updates.
    set +e
    sudo -u vscode bash -c 'export PATH="/home/vscode/.local/share/pnpm:$PATH" && pnpm add -g @qwen-code/qwen-code@latest --config.global-bin-dir=/home/vscode/.local/share/pnpm' || echo "  ⚠️ pnpm install/update failed, continuing..."
    set -e
fi

# 4. Fix workspace permissions if mounted by Docker
# (Often owned by root initially in binds)
# WARNING: Recursive chown on /workspaces (the entire repo) causes massive I/O spikes and can crash WSL
if [ -d "/workspaces" ] && [ ! -w "/workspaces" ]; then
    echo "🔧 Fixing permissions on /workspaces (non-recursive)..."
    sudo chown vscode:vscode /workspaces
fi

echo "✅ Entrypoint setup complete. Starting services..."

# 5. Start SSH Daemon
echo "Starting SSH Daemon..."
sudo /usr/sbin/sshd -D -e
