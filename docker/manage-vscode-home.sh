#!/bin/bash
# Manage persisted /home/vscode volume for container resets
# Usage: ./manage-vscode-home.sh [backup|restore|reset|info]

set -e

COMPOSE_FILE="docker/docker-compose.yml"
VOLUME_NAME="ocr-dev_vscode_home"
BACKUP_DIR="docker/backups"
CONTAINER_NAME="ocr-recsys-dev"

cmd_info() {
    echo "=== VS Code Home Volume Info ==="
    echo ""

    # Check if volume exists
    if docker volume inspect "$VOLUME_NAME" > /dev/null 2>&1; then
        echo "✅ Volume exists: $VOLUME_NAME"
        echo ""
        echo "Volume Details:"
        docker volume inspect "$VOLUME_NAME" | jq -r '.[0] | "  Driver: \(.Driver)\n  Mountpoint: \(.Mountpoint)"'
        echo ""

        # Show size if container is running
        if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
            echo "Volume Size:"
            docker exec "$CONTAINER_NAME" du -sh /home/vscode 2>/dev/null || echo "  (unable to get size)"
            echo ""

            echo "Key Directories:"
            docker exec "$CONTAINER_NAME" bash -c '
                for dir in .vscode-server .cache .local .config .pyenv .npm .bashrc; do
                    if [ -e /home/vscode/$dir ]; then
                        size=$(du -sh /home/vscode/$dir 2>/dev/null | cut -f1)
                        echo "  $dir: $size"
                    fi
                done
            ' 2>/dev/null || echo "  (unable to list directories)"
        else
            echo "⚠️  Container not running - start it to see usage details"
        fi
    else
        echo "❌ Volume does not exist: $VOLUME_NAME"
        echo ""
        echo "The volume will be created automatically when you start the container."
    fi
}

cmd_backup() {
    echo "=== Backing up /home/vscode ==="

    if ! docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
        echo "❌ Container is not running. Start it first:"
        echo "   cd docker && docker compose up -d"
        exit 1
    fi

    mkdir -p "$BACKUP_DIR"
    BACKUP_FILE="$BACKUP_DIR/vscode_home_$(date +%Y%m%d_%H%M%S).tar.gz"

    echo "📦 Creating backup..."
    echo "   Source: /home/vscode (in container)"
    echo "   Destination: $BACKUP_FILE"
    echo ""

    # Backup excluding mounted directories and large caches
    docker exec "$CONTAINER_NAME" tar czf - \
        --exclude='.ssh-mount' \
        --exclude='.cache/uv/builds-*' \
        --exclude='.cache/pip/http' \
        --exclude='.npm/_cacache' \
        -C /home vscode \
        > "$BACKUP_FILE"

    SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    echo ""
    echo "✅ Backup complete: $BACKUP_FILE ($SIZE)"
    echo ""
    echo "To restore this backup later:"
    echo "   ./manage-vscode-home.sh restore $BACKUP_FILE"
}

cmd_restore() {
    local backup_file="$1"

    if [ -z "$backup_file" ]; then
        echo "❌ Please specify a backup file to restore"
        echo ""
        echo "Available backups:"
        ls -lh "$BACKUP_DIR"/*.tar.gz 2>/dev/null || echo "   (no backups found in $BACKUP_DIR)"
        exit 1
    fi

    if [ ! -f "$backup_file" ]; then
        echo "❌ Backup file not found: $backup_file"
        exit 1
    fi

    echo "=== Restoring /home/vscode ==="
    echo "⚠️  WARNING: This will overwrite current /home/vscode contents!"
    echo ""
    read -p "Continue? (yes/no): " confirm

    if [ "$confirm" != "yes" ]; then
        echo "Cancelled."
        exit 0
    fi

    if ! docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
        echo "❌ Container is not running. Start it first:"
        echo "   cd docker && docker compose up -d"
        exit 1
    fi

    echo ""
    echo "📦 Restoring from: $backup_file"

    # Extract backup into container
    docker exec -i "$CONTAINER_NAME" tar xzf - -C /home < "$backup_file"

    # Fix ownership
    docker exec "$CONTAINER_NAME" sudo chown -R vscode:vscode /home/vscode

    echo ""
    echo "✅ Restore complete!"
    echo ""
    echo "⚠️  You may need to restart the container for all changes to take effect:"
    echo "   cd docker && docker compose restart"
}

cmd_reset() {
    echo "=== Reset /home/vscode Volume ==="
    echo "⚠️  WARNING: This will DELETE all persisted data!"
    echo ""
    echo "This includes:"
    echo "  • VS Code Server state"
    echo "  • Copilot/Antigravity chat history"
    echo "  • Shell history (.bash_history)"
    echo "  • Cached dependencies (.cache, .npm, .pyenv)"
    echo "  • User configurations (.config, .local)"
    echo ""
    read -p "Are you sure? Type 'DELETE' to confirm: " confirm

    if [ "$confirm" != "DELETE" ]; then
        echo "Cancelled."
        exit 0
    fi

    echo ""
    echo "Stopping container..."
    cd docker && docker compose down

    echo "Removing volume..."
    docker volume rm "$VOLUME_NAME" 2>/dev/null || echo "  (volume already removed)"

    echo ""
    echo "✅ Volume reset complete!"
    echo ""
    echo "Start the container to create a fresh /home/vscode:"
    echo "   cd docker && docker compose up -d"
}

cmd_clone() {
    echo "=== Clone /home/vscode to host ==="
    echo "This creates a one-time copy for inspection or migration."
    echo ""

    if ! docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
        echo "❌ Container is not running. Start it first:"
        echo "   cd docker && docker compose up -d"
        exit 1
    fi

    CLONE_DIR="docker/vscode_home_clone_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$CLONE_DIR"

    echo "📂 Cloning to: $CLONE_DIR"

    docker cp "$CONTAINER_NAME:/home/vscode/." "$CLONE_DIR/"

    echo ""
    echo "✅ Clone complete!"
    echo ""
    echo "You can now inspect or modify the contents:"
    echo "   cd $CLONE_DIR"
}

# Main
case "${1:-info}" in
    info)
        cmd_info
        ;;
    backup)
        cmd_backup
        ;;
    restore)
        cmd_restore "$2"
        ;;
    reset)
        cmd_reset
        ;;
    clone)
        cmd_clone
        ;;
    *)
        echo "Usage: $0 {info|backup|restore|reset|clone}"
        echo ""
        echo "Commands:"
        echo "  info     - Show volume information and usage"
        echo "  backup   - Create a backup of /home/vscode"
        echo "  restore  - Restore from a backup file"
        echo "  reset    - Delete the volume and start fresh"
        echo "  clone    - Copy volume contents to host for inspection"
        exit 1
        ;;
esac
