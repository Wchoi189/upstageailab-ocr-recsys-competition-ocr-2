#!/bin/bash
set -e

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
# Destination: Your workspace
SOURCE_ROOT="/mnt/external_artifacts"
DEST_ROOT="/workspaces"
FOLDERS=("apps" "archive" "data" "outputs" "packages")

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
    # We do this carefully.
    for old in "extensions" "hydra_outputs" "lightning_logs" "wandb"; do
        if [ -L "$DEST_ROOT/$old" ]; then
             echo "  🧹 Cleaning up deprecated symlink: $old"
             rm "$DEST_ROOT/$old"
        fi
    done

    # Nested Symlinks for convenience (optional, but requested by user config implied?)
    # paths.yaml now uses ${output_dir}/wandb, so /workspaces/outputs/wandb is valid.
    # No extra symlinks needed if the migration worked and 'outputs' is linked.

    echo "🔗 Symlink setup complete!"
else
    echo "⚠️  External artifacts directory not found at $SOURCE_ROOT"
    echo "   Skipping symlink creation."
fi

# 4. Fix workspace permissions if mounted by Docker
# (Often owned by root initially in binds)
if [ -d "/workspaces" ]; then
    sudo chown -R vscode:vscode /workspaces
fi

# 5. Start SSH Daemon
echo "Starting SSH Daemon..."
sudo /usr/sbin/sshd -D -e
