#!/bin/bash
# Script to test different scenarios for finding the freeze point
#
# NOTE: The Streamlit freeze issue was resolved in October 2025 (see docs/sessions/2025-10-21/issue-resolved.md).
# This script is kept for future debugging purposes.

set -e

echo "=========================================="
echo "STREAMLIT FREEZE DEBUG - SCENARIO TESTER"
echo "=========================================="
echo ""

show_menu() {
    echo "Choose a test scenario:"
    echo ""
    echo "  1) Run in FOREGROUND mode (recommended first)"
    echo "     - Shows all print statements in terminal"
    echo "     - You'll see exactly where it freezes"
    echo ""
    echo "  2) Use MINIMAL results renderer"
    echo "     - Disables st.dataframe() and st.expander()"
    echo "     - Only shows simple text"
    echo "     - Tests if freeze is widget-related"
    echo ""
    echo "  3) Run ORIGINAL version in background"
    echo "     - Normal operation with full logging"
    echo "     - Check logs after freeze"
    echo ""
    echo "  4) RESTORE original results.py"
    echo "     - Undo minimal renderer changes"
    echo ""
    echo "  0) Exit"
    echo ""
    echo -n "Enter choice: "
}

run_foreground() {
    echo ""
    echo "=========================================="
    echo "RUNNING IN FOREGROUND MODE"
    echo "=========================================="
    echo ""
    echo "Instructions:"
    echo "1. The app will start in this terminal"
    echo "2. Open browser to: http://localhost:8504"
    echo "3. Upload an image and run inference"
    echo "4. WATCH THIS TERMINAL for print statements"
    echo "5. Note the LAST message before freeze"
    echo "6. Press Ctrl+C to stop"
    echo ""
    echo "Starting in 3 seconds..."
    sleep 3

    # Stop background process if running
    make stop-inference-ui 2>/dev/null || true

    # Run in foreground
    cd ui/apps/inference
    uv run streamlit run app.py --server.port=8504
}

use_minimal_renderer() {
    echo ""
    echo "=========================================="
    echo "SWITCHING TO MINIMAL RENDERER"
    echo "=========================================="
    echo ""

    # Backup original if not already backed up
    if [ ! -f "ui/apps/inference/components/results.py.BACKUP" ]; then
        echo "Creating backup of original results.py..."
        cp ui/apps/inference/components/results.py ui/apps/inference/components/results.py.BACKUP
    fi

    # Check if minimal renderer exists
    if [ ! -f "ui/apps/inference/components/results_minimal.py" ]; then
        echo "ERROR: results_minimal.py not found!"
        echo "Please run the main script first to generate it."
        return 1
    fi

    # Copy minimal renderer over results.py
    echo "Replacing results.py with minimal version..."
    cp ui/apps/inference/components/results_minimal.py ui/apps/inference/components/results.py

    echo ""
    echo "✓ Minimal renderer installed"
    echo ""
    echo "Now run the app (choose option 1 or 3)"
    echo "The results display will be text-only with no dataframe/expander"
    echo ""
}

run_background() {
    echo ""
    echo "=========================================="
    echo "RUNNING IN BACKGROUND MODE"
    echo "=========================================="
    echo ""

    make stop-inference-ui 2>/dev/null || true
    make ui-infer

    echo ""
    echo "App started in background"
    echo "URL: http://localhost:8504"
    echo ""
    echo "To watch logs in real-time, run in another terminal:"
    echo "  tail -f logs/ui/inference_8504.err"
    echo ""
    echo "After freeze, check logs with:"
    echo "  bash scripts/troubleshooting/check_streamlit_logs.sh"
    echo ""
}

restore_original() {
    echo ""
    echo "=========================================="
    echo "RESTORING ORIGINAL RESULTS.PY"
    echo "=========================================="
    echo ""

    if [ -f "ui/apps/inference/components/results.py.BACKUP" ]; then
        echo "Restoring from backup..."
        cp ui/apps/inference/components/results.py.BACKUP ui/apps/inference/components/results.py
        echo "✓ Original results.py restored"
    else
        echo "WARNING: No backup found (results.py.BACKUP)"
        echo "The original version should already be in use"
    fi
    echo ""
}

# Main loop
while true; do
    show_menu
    read -r choice

    case $choice in
        1)
            run_foreground
            ;;
        2)
            use_minimal_renderer
            ;;
        3)
            run_background
            ;;
        4)
            restore_original
            ;;
        0)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            echo ""
            ;;
    esac

    # After running foreground mode, we'll be back here
    if [ "$choice" != "1" ]; then
        echo ""
        echo "Press Enter to continue..."
        read -r
    fi
done
