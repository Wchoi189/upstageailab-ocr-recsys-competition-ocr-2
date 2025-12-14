# ScanTailor Advanced Offscreen Implementation - Results

**Date**: 2025-11-22
**Status**: Partially Working - Requires Project File Format Investigation

## Summary

Implemented offscreen wrapper for ScanTailor Advanced. Offscreen plugin works, but application requires project files (`.st` format) rather than direct image processing.

## Implementation

### Created Files

1. **`scripts/test_scantailor_offscreen.py`** - Python wrapper using Qt offscreen plugin
   - Checks for ScanTailor Advanced and offscreen plugin
   - Sets `QT_QPA_PLATFORM=offscreen` environment variable
   - Attempts to process images

### Findings

#### ✅ What Works

1. **Qt Offscreen Plugin**: Available and functional
   - Location: `/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/libqoffscreen.so`
   - Application starts without X11 errors
   - No display connection required

2. **Command-Line Arguments**: Application accepts project file path
   ```cpp
   if (args.size() > 1) {
     mainWnd->openProject(args.at(1));
   }
   ```

#### ❌ What Doesn't Work

1. **Direct Image Processing**: Application doesn't accept image files directly
   - Passing image path causes timeout (waits for GUI interaction)
   - Requires `.st` project file format

2. **Automatic Processing**: Application is designed for interactive use
   - Opens project and waits for user interaction
   - No built-in batch processing mode
   - Event loop may not process correctly in offscreen mode without proper setup

3. **Project File Format**: Unknown structure
   - Need to investigate `.st` file format
   - May require reverse-engineering from existing projects

## Test Results

```bash
$ python scripts/test_scantailor_offscreen.py --input-dir outputs/perspective_test --num-samples 1
✓ ScanTailor Advanced found
✓ Qt offscreen plugin available
Processing 1 images
Running (offscreen): scantailor <image_path>
✗ ScanTailor timed out after 300s
```

**Issue**: Application hangs when given image file path (expects project file).

## Next Steps

### Option 1: Investigate Project File Format (Recommended)

**Approach**:
1. Create a test project in GUI mode (if X11 available)
2. Examine `.st` file structure
3. Create Python generator for project files
4. Test processing with generated project files

**Challenges**:
- Need access to GUI to create sample project
- File format may be binary or complex XML
- May require understanding internal data structures

### Option 2: Use Xvfb for Full GUI Automation

**Approach**:
1. Use Xvfb (virtual framebuffer) instead of offscreen plugin
2. Automate GUI interactions with tools like `xdotool` or `pyautogui`
3. Create project, configure settings, process images programmatically

**Pros**:
- Full GUI functionality available
- Can use all features

**Cons**:
- More complex setup
- Slower (full GUI overhead)
- Requires GUI automation libraries

### Option 3: Alternative Tools

**Consider**:
- Original ScanTailor CLI (blocked by Qt4 dependency)
- Other document processing tools with CLI support
- Custom implementation using OpenCV/image processing libraries

## Code Status

**Current Implementation**: `scripts/test_scantailor_offscreen.py`
- ✅ Prerequisites checking
- ✅ Offscreen environment setup
- ✅ Basic command execution
- ❌ Project file creation (placeholder)
- ❌ Automatic processing trigger

## Conclusion

The offscreen approach is **technically feasible** but requires:
1. Understanding project file format (`.st` files)
2. Ability to trigger processing programmatically
3. Possibly GUI automation for full functionality

**Recommendation**:
- Short-term: Investigate project file format if GUI access available
- Alternative: Consider if ScanTailor is the right tool for this use case, or if simpler image processing libraries would suffice

## References

- Implementation: `scripts/test_scantailor_offscreen.py`
- Offscreen Assessment: `docs/assessments/scantailor-gui-api-repurpose-assessment.md`
- Original CLI Attempt: `docs/assessments/scantailor-original-cli-installation-results.md`

