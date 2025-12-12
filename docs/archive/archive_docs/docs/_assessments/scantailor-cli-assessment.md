# ScanTailor CLI Capability Assessment

**Date**: 2025-11-22
**Status**: Assessment Complete

## Executive Summary

**Current Situation**: ScanTailor Advanced (installed) is GUI-only. The original ScanTailor has CLI support but is archived (unmaintained since Nov 2020). The `scantailor-libs-build` repository is a build helper for dependencies, not a CLI solution.

**Recommendation**: Use original ScanTailor (Option 1) for CLI capabilities, despite being older. The CLI functionality is stable and sufficient for batch processing.

## Repository Assessment

### scantailor-libs-build Repository

**URL**: https://github.com/4lex4/scantailor-libs-build.git

**Purpose**: Build helper for compiling ScanTailor Advanced and its dependencies (Qt, Boost, libpng, etc.)

**Key Findings**:
- ✅ Provides comprehensive build instructions for Windows, Linux, macOS
- ✅ Handles dependency compilation (Qt, Boost, image libraries)
- ❌ **Does NOT provide CLI functionality** - only builds the GUI application
- ❌ Same limitation as ScanTailor Advanced: GUI-only output

**Verdict**: Not suitable for CLI use case. This repository only helps build the GUI version.

### Original ScanTailor (Option 1)

**Repository**: https://github.com/scantailor/scantailor
**Status**: Archived (unmaintained since November 29, 2020)
**Latest Version**: 0.9.9.2 (RELEASE_0_9_9_2)

**Key Features**:
- ✅ **Has CLI version**: `scantailor-cli` command-line tool
- ✅ Headless operation (no GUI required)
- ✅ Batch processing support
- ⚠️ Older codebase (last updated 2020)
- ⚠️ May have compatibility issues with newer systems

**CLI Capabilities**:
```bash
scantailor-cli --help
scantailor-cli --margins=0.1 --alignment=auto input.jpg output/
```

### ScanTailor Advanced (Currently Installed)

**Repository**: https://github.com/4lex4/scantailor-advanced
**Status**: Actively maintained
**Latest Version**: Current (2024+)

**Key Features**:
- ✅ Modern, actively maintained
- ✅ Enhanced features and bug fixes
- ❌ **GUI-only** - no CLI version available
- ❌ Requires X11 display for headless operation

## Comparison Matrix

| Feature | Original ScanTailor | ScanTailor Advanced | scantailor-libs-build |
|---------|-------------------|-------------------|----------------------|
| CLI Support | ✅ Yes (`scantailor-cli`) | ❌ No | ❌ No |
| Headless Operation | ✅ Yes | ❌ No (requires X11) | ❌ No |
| Maintenance Status | ⚠️ Archived (2020) | ✅ Active | ✅ Active |
| Build Helper | ❌ No | ❌ No | ✅ Yes |
| Batch Processing | ✅ Yes | ❌ Manual only | N/A |
| Use Case Fit | ✅ **Best for CLI** | ❌ GUI only | ❌ Build tool only |

## Recommendations

### Option A: Use Original ScanTailor (Recommended for CLI)

**Pros**:
- Has working CLI (`scantailor-cli`)
- Headless operation
- Stable, proven codebase
- Sufficient for batch processing needs

**Cons**:
- Older codebase (2020)
- No active maintenance
- May need compatibility fixes for newer systems

**Installation**:
```bash
# Clone original ScanTailor
git clone https://github.com/scantailor/scantailor.git
cd scantailor

# Build (same prerequisites as ScanTailor Advanced)
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

# Verify CLI
scantailor-cli --help
```

### Option B: Use X11 Virtual Display with ScanTailor Advanced

**Pros**:
- Modern, maintained codebase
- Enhanced features

**Cons**:
- Requires Xvfb or X11 forwarding
- More complex setup
- Slower (GUI overhead)
- Not true headless operation

**Setup**:
```bash
# Install Xvfb
sudo apt-get install xvfb

# Run with virtual display
xvfb-run -a scantailor --batch input.jpg output/
```

### Option C: Alternative Tools

**Spreads** (https://spreads.readthedocs.io/):
- Modern CLI tool for scanned document processing
- Actively maintained
- Different workflow (YAML config-based)
- May require workflow adaptation

## Decision Matrix

**If you need**:
- ✅ **CLI/headless operation** → Use **Original ScanTailor** (Option A)
- ✅ **Modern features + GUI** → Keep **ScanTailor Advanced** (current)
- ✅ **Build help only** → Use **scantailor-libs-build** (not for CLI)

## Implementation Plan

### For CLI Use Case (Recommended)

1. **Install Original ScanTailor**:
   ```bash
   git clone https://github.com/scantailor/scantailor.git
   cd scantailor
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   sudo make install
   ```

2. **Update Integration Script**:
   - Modify `scripts/test_scantailor_integration.py` to use `scantailor-cli`
   - Remove X11 dependencies
   - Test headless operation

3. **Verify CLI Functionality**:
   ```bash
   scantailor-cli --help
   python scripts/test_scantailor_integration.py --input-dir test/ --output-dir output/
   ```

## Conclusion

The `scantailor-libs-build` repository is **not suitable** for CLI use cases - it only provides build instructions for the GUI version. For CLI capabilities, **Option 1 (Original ScanTailor)** is the recommended path despite being older, as it provides the required `scantailor-cli` tool for headless batch processing.

## References

- Original ScanTailor: https://github.com/scantailor/scantailor
- ScanTailor Advanced: https://github.com/4lex4/scantailor-advanced
- scantailor-libs-build: https://github.com/4lex4/scantailor-libs-build
- Installation Guide: `scripts/SCANTAILOR_INSTALL.md`

