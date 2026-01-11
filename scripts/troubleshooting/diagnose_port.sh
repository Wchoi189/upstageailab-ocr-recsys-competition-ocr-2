#!/bin/bash
# Diagnostic script for port 8000 issues

PORT=8000
echo "=== Port 8000 Diagnostic ==="
echo ""

echo "1. Checking for processes using port $PORT:"
if command -v lsof >/dev/null 2>&1; then
    lsof -i :$PORT || echo "  No processes found with lsof"
else
    echo "  lsof not available"
fi

echo ""
echo "2. Checking for uvicorn processes:"
pgrep -af uvicorn || echo "  No uvicorn processes found"

echo ""
echo "3. Checking for mkdocs processes (also uses port 8000):"
pgrep -af mkdocs || echo "  No mkdocs processes found"

echo ""
echo "4. Checking socket states:"
ss -tan | grep :$PORT || echo "  No sockets found in ss output"

echo ""
echo "5. Checking for any Python processes that might be holding the port:"
ps aux | grep -E "python.*8000|uvicorn|mkdocs" | grep -v grep || echo "  No matching Python processes"

echo ""
echo "6. Testing port binding:"
uv run python3 << 'EOF'
import socket
import sys

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('127.0.0.1', 8000))
    s.close()
    print("  ✓ Port 8000 is available for binding")
except OSError as e:
    print(f"  ✗ Port 8000 is NOT available: {e}")
    sys.exit(1)
EOF

echo ""
echo "=== End Diagnostic ==="

