# Original ScanTailor CLI Installation Attempt - Results

**Date**: 2025-11-22
**Status**: Blocked - Qt4 Dependency Issue

## Summary

Attempted to install original ScanTailor for CLI support, but encountered blocking dependency issues.

## Findings

### 1. Repository Structure
- **Repository**: https://github.com/scantailor/scantailor
- **Latest Release**: RELEASE_0_9_9_2 (checked out)
- **Build System**: CMake

### 2. Critical Issue: Qt4 Dependency

**Problem**: Original ScanTailor requires Qt4, which is:
- ❌ Not available in Ubuntu 22.04 repositories
- ❌ Deprecated and unmaintained
- ❌ Incompatible with Qt5 (which is installed)

**CMake Error**:
```
Found unsuitable Qt version "5.15.3" from /usr/bin/qmake
CMake Error: Qt4 could not be found.
```

**CMakeLists.txt Requirements**:
```cmake
INCLUDE(FindQt4)
IF(NOT QT4_FOUND)
    FATAL_ERROR "Qt4 could not be found."
ENDIF(NOT QT4_FOUND)
```

### 3. No CLI Executable in Source

**Finding**: The original ScanTailor source code only builds a GUI executable:
- `ADD_EXECUTABLE(scantailor ...)` - Single GUI application
- No separate `scantailor-cli` target
- No CLI-specific source files found

**Conclusion**: The "scantailor-cli" mentioned in some documentation may refer to:
- A separate package that doesn't exist in current repositories
- Outdated information
- A different fork/branch

### 4. Package Availability

**Tested**:
```bash
apt-cache search scantailor        # No results
apt-cache show scantailor-cli     # Package not found
```

**Result**: No pre-built packages available for Ubuntu 22.04.

## Options Moving Forward

### Option A: Install Qt4 (Not Recommended)

**Challenges**:
- Qt4 is deprecated and unsupported
- Would require building from source or finding old repositories
- Security concerns (unmaintained)
- Complex dependency resolution

**Verdict**: ❌ Not practical for production use

### Option B: Use ScanTailor Advanced with Offscreen Mode (Recommended)

**Advantages**:
- ✅ Already installed and working
- ✅ Modern, maintained codebase
- ✅ Qt5 compatible
- ✅ Qt offscreen plugin available: `/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/libqoffscreen.so`
- ✅ Can be used headlessly

**Implementation**: Use `QT_QPA_PLATFORM=offscreen` environment variable

**Status**: ✅ Ready to implement

### Option C: Port Original ScanTailor to Qt5

**Effort**: High
- Would require modifying CMakeLists.txt
- Updating Qt4 API calls to Qt5
- Testing and debugging
- Maintenance burden

**Verdict**: ⚠️ Significant development effort, not justified

## Recommendation

**Use ScanTailor Advanced with Qt Offscreen Plugin**

Since:
1. Original ScanTailor CLI is not practically buildable on modern systems
2. ScanTailor Advanced is already installed
3. Qt offscreen plugin is available and tested
4. Internal processing APIs exist and can be accessed

**Next Steps**:
1. Implement Python wrapper using offscreen mode
2. Investigate project file format (`.st` files)
3. Create batch processing interface

## References

- Original ScanTailor: https://github.com/scantailor/scantailor (archived)
- ScanTailor Advanced: https://github.com/4lex4/scantailor-advanced
- Qt Offscreen Plugin Assessment: `docs/assessments/scantailor-gui-api-repurpose-assessment.md`

