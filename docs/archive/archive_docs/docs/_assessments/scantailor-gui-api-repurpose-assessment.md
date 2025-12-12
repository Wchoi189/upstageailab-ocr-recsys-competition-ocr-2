# ScanTailor Advanced: GUI API Repurposing Assessment

**Date**: 2025-11-22
**Status**: Assessment Complete

## Executive Summary

**Question**: Can we repurpose GUI functionality through intermediate CLI-like API endpoints?

**Answer**: **Partially Yes** - ScanTailor Advanced has internal processing APIs, but they require Qt GUI framework. We can use Qt's offscreen platform plugin for headless operation, but this requires code modifications or wrapper creation.

## Architecture Analysis

### Current Structure

ScanTailor Advanced follows a layered architecture:

```
GUI Layer (MainWindow)
    ↓
Core Processing Layer (Project, Filters, Tasks)
    ↓
Image Processing Library (imageproc/)
```

### Key Findings

1. **Command-Line Arguments**: The `main.cpp` accepts project file paths:
   ```cpp
   if (args.size() > 1) {
     mainWnd->openProject(args.at(1));
   }
   ```

2. **Internal Processing APIs**: Core processing is separated from GUI:
   - `Project` class handles project management
   - `Filter` classes (deskew, fix_orientation, output, etc.) handle processing
   - `Task` classes execute processing operations
   - `BackgroundTask` system for async processing

3. **Qt Platform Plugins**: The application supports multiple Qt platforms:
   - `xcb` (X11 - requires display)
   - `offscreen` (headless rendering)
   - `minimal` (minimal GUI)
   - `eglfs` (embedded)

4. **Project File Format**: Uses `.st` project files that can be created programmatically

## Repurposing Options

### Option 1: Qt Offscreen Platform Plugin (Recommended)

**Approach**: Use Qt's offscreen platform plugin to run GUI application headlessly.

**Pros**:
- ✅ No code modifications needed
- ✅ Uses existing internal APIs
- ✅ Full feature access
- ✅ Can process project files

**Cons**:
- ⚠️ Requires Qt offscreen plugin (usually available)
- ⚠️ Still launches full Qt application (memory overhead)
- ⚠️ Slower than pure CLI (GUI framework overhead)

**Implementation**:
```bash
# Set Qt platform to offscreen
export QT_QPA_PLATFORM=offscreen

# Run ScanTailor with project file
scantailor project.st
```

**Code Example**:
```python
import subprocess
import os

def process_with_scantailor_offscreen(project_file: str):
    env = os.environ.copy()
    env['QT_QPA_PLATFORM'] = 'offscreen'

    result = subprocess.run(
        ['scantailor', project_file],
        env=env,
        capture_output=True,
        timeout=300
    )
    return result.returncode == 0
```

### Option 2: Create Minimal CLI Wrapper

**Approach**: Create a new executable that uses ScanTailor's core libraries without GUI.

**Pros**:
- ✅ True headless operation
- ✅ Lower memory footprint
- ✅ Faster startup
- ✅ Can expose CLI interface

**Cons**:
- ❌ Requires C++ development
- ❌ Need to understand internal APIs
- ❌ Maintenance burden
- ❌ May miss GUI-specific features

**Implementation** (conceptual):
```cpp
// scantailor-cli-wrapper.cpp
#include <core/Project.h>
#include <core/filters/output/Filter.h>
#include <QCoreApplication>  // Not QApplication!

int main(int argc, char* argv[]) {
    QCoreApplication app(argc, argv);

    // Load project
    Project project;
    project.load(argv[1]);

    // Process all pages
    project.processAllPages();

    // Export results
    project.exportOutput(argv[2]);

    return 0;
}
```

### Option 3: Python C++ Bindings

**Approach**: Create Python bindings for ScanTailor's core processing classes.

**Pros**:
- ✅ Native Python integration
- ✅ Full API access
- ✅ Flexible usage

**Cons**:
- ❌ Complex implementation (pybind11, SWIG, etc.)
- ❌ Significant development effort
- ❌ Maintenance overhead

### Option 4: Project File Automation

**Approach**: Generate `.st` project files programmatically and process them.

**Pros**:
- ✅ Uses existing application
- ✅ No code modifications
- ✅ Can batch process

**Cons**:
- ⚠️ Requires understanding `.st` file format
- ⚠️ Still needs GUI or offscreen mode
- ⚠️ Less flexible than direct API

## Testing Qt Offscreen Mode

### Verification Steps

1. **Check Platform Plugin Availability**:
   ```bash
   ls /usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/libqoffscreen.so
   ```

2. **Test Offscreen Execution**:
   ```bash
   QT_QPA_PLATFORM=offscreen scantailor --version
   ```

3. **Process Project File**:
   ```bash
   QT_QPA_PLATFORM=offscreen scantailor project.st
   ```

## Comparison with Original ScanTailor CLI

| Feature | Original CLI | Advanced Offscreen | Advanced Wrapper |
|---------|-------------|-------------------|------------------|
| Headless | ✅ Native | ✅ Via plugin | ✅ Native |
| Startup Time | Fast | Medium | Fast |
| Memory Usage | Low | Medium | Low |
| Feature Access | Limited | Full | Full |
| Maintenance | None | Low | High |
| Implementation | Existing | Easy | Complex |

## Recommendations

### Short-term Solution: Qt Offscreen Plugin

**Action Items**:
1. Test Qt offscreen plugin availability
2. Create wrapper script using `QT_QPA_PLATFORM=offscreen`
3. Generate project files programmatically
4. Process images via project files

**Code Example**:
```python
# scripts/scantailor_offscreen_wrapper.py
import subprocess
import tempfile
from pathlib import Path

def process_image_offscreen(input_image: Path, output_dir: Path):
    """Process image using ScanTailor Advanced in offscreen mode."""
    # Create project file (format needs investigation)
    project_file = create_project_file(input_image, output_dir)

    env = os.environ.copy()
    env['QT_QPA_PLATFORM'] = 'offscreen'

    result = subprocess.run(
        ['scantailor', str(project_file)],
        env=env,
        capture_output=True,
        timeout=300
    )

    return result.returncode == 0
```

### Long-term Solution: Evaluate Original ScanTailor CLI

If offscreen mode doesn't meet performance requirements, consider:
1. Installing original ScanTailor for CLI access
2. Creating hybrid approach (use CLI when available, offscreen as fallback)

## Project File Format Investigation

The `.st` project file format needs investigation to enable programmatic creation:

1. **File Structure**: XML-based format (likely)
2. **Required Fields**: Image paths, processing settings, output paths
3. **Documentation**: May need to reverse-engineer from existing projects

**Next Steps**:
- Create a test project in GUI
- Examine `.st` file structure
- Create Python generator for project files

## Conclusion

**Yes, GUI functionality can be repurposed**, but with caveats:

1. **Easiest**: Use Qt offscreen platform plugin (no code changes)
2. **Best Performance**: Create minimal CLI wrapper (requires development)
3. **Most Flexible**: Python bindings (significant effort)

**Recommended Path**: Start with Qt offscreen plugin to validate feasibility, then decide if wrapper development is justified based on performance needs.

## References

- ScanTailor Advanced Source: `/workspaces/.../scantailor-advanced/`
- Qt Platform Plugins: https://doc.qt.io/qt-5/qt.html#ApplicationType-enum
- GitHub Issue #171: CLI support request
- Original ScanTailor CLI: https://github.com/scantailor/scantailor

