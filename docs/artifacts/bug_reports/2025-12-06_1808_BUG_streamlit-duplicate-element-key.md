---
title: "Bug 2025 012 Streamlit Duplicate Element Key"
date: "2025-12-06 18:08 (KST)"
type: "bug_report"
category: "troubleshooting"
status: "active"
version: "1.0"
tags: ['bug_report', 'troubleshooting']
---



# BUG-2025-012: Streamlit Duplicate Element Key in Unified OCR App

**Date:** October 21, 2025
**Bug ID:** BUG-2025-012
**Status:** ✅ **FIXED**
**Severity:** High (App Crash)
**Component:** Unified OCR App - Inference Mode

---

## Executive Summary

**Issue**: `StreamlitDuplicateElementKey` exception when accessing the inference mode of the unified OCR app, preventing the app from loading.

**Root Cause**: Widget key `"mode_selector"` was used in two different contexts:
1. Main app mode selector (preprocessing/inference/comparison) in `app.py:93`
2. Inference processing mode selector (single/batch) in `checkpoint_selector.py:186`

**Fix**: Renamed the inference processing mode selector key to `"inference_processing_mode_selector"` to ensure uniqueness.

**Impact**: Inference mode now loads correctly without key conflicts. All three app modes (preprocessing, inference, comparison) are now fully functional.

---

## Error Details

### Stack Trace

```
StreamlitDuplicateElementKey: There are multiple elements with the same
`key='mode_selector'`. To fix this, please make sure that the `key` argument is unique
for each element you create.

  File "ui/apps/unified_ocr_app/app.py", line 122, in main
    render_inference_mode(state, config, mode_config)

  File "ui/apps/unified_ocr_app/components/inference/checkpoint_selector.py", line 181
    selected_display = st.radio(
        label,
        options=list(display_options.keys()),
        index=default_index,
        horizontal=True,
        key="mode_selector",  # ❌ DUPLICATE KEY
    )
```

### Error Location

**File**: `streamlit-unified-app.log`
**Timestamp**: 2025-10-21 13:54:13

---

## Root Cause Analysis

### Key Conflict Locations

1. **Main Mode Selector** (`app.py:93`)
   ```python
   mode = st.radio(
       "Select Mode",
       options=list(mode_icons.keys()),
       key="mode_selector",  # First usage
   )
   ```

2. **Inference Processing Mode Selector** (`checkpoint_selector.py:186`)
   ```python
   selected_display = st.radio(
       label,
       options=list(display_options.keys()),
       key="mode_selector",  # ❌ Duplicate usage
   )
   ```

### Why This Happened

During Phase 4 (Inference Mode) implementation, the processing mode selector was created with a generic key name `"mode_selector"` without considering that the same key was already in use in the main app for the top-level mode selection.

Streamlit requires all widget keys to be globally unique within the app session.

---

## Fix Implementation

### Changes Made

**File**: `ui/apps/unified_ocr_app/components/inference/checkpoint_selector.py`
**Line**: 186

```python
# Before (❌ Duplicate Key)
selected_display = st.radio(
    label,
    options=list(display_options.keys()),
    index=default_index,
    horizontal=True,
    key="mode_selector",
)

# After (✅ Unique Key)
selected_display = st.radio(
    label,
    options=list(display_options.keys()),
    index=default_index,
    horizontal=True,
    key="inference_processing_mode_selector",
)
```

### Key Naming Convention

To prevent future conflicts, the fix uses a descriptive, scoped key name:
- `inference_` - Component scope
- `processing_mode_` - Functionality
- `selector` - Widget type

This naming pattern should be followed for all Streamlit widget keys in the unified app.

---

## Testing

### Verification Steps

1. ✅ Start unified OCR app
2. ✅ Navigate to inference mode
3. ✅ Verify processing mode selector renders correctly
4. ✅ Switch between single and batch modes
5. ✅ Verify no key conflict errors

### Test Results

```bash
# App starts successfully
uv run streamlit run ui/apps/unified_ocr_app/app.py

# No StreamlitDuplicateElementKey errors
# All modes accessible: preprocessing, inference, comparison
```

---

## Impact Assessment

### Before Fix
- ❌ Inference mode completely inaccessible
- ❌ App crashes with `StreamlitDuplicateElementKey` exception
- ❌ Phase 4 and Phase 6 functionality unavailable

### After Fix
- ✅ Inference mode loads correctly
- ✅ Processing mode selector works as intended
- ✅ All three app modes fully functional
- ✅ No key conflicts

---

## Prevention Guidelines

### Widget Key Best Practices

1. **Use Descriptive Keys**: Always include component scope and functionality
   - Good: `"inference_processing_mode_selector"`
   - Bad: `"mode_selector"`, `"selector"`, `"radio1"`

2. **Scope Keys by Component**: Prefix keys with component/mode name
   - Preprocessing: `"preprocessing_*"`
   - Inference: `"inference_*"`
   - Comparison: `"comparison_*"`

3. **Check for Conflicts**: Before adding a widget key, search the codebase
   ```bash
   grep -r 'key="your_key_name"' ui/apps/unified_ocr_app/
   ```

4. **Document Key Usage**: Add comments for non-obvious keys
   ```python
   st.radio(
       "Mode",
       options=modes,
       key="inference_processing_mode_selector",  # Scoped to avoid conflict with main mode selector
   )
   ```

---

## Related Issues

- **Phase 4**: Inference Mode implementation (where duplicate key was introduced)
- **Phase 7**: Documentation phase (where bug was discovered and fixed)

---

## Resolution

**Status**: ✅ **FIXED**
**Fixed By**: Key rename to `"inference_processing_mode_selector"`
**Date Fixed**: October 21, 2025
**Verified**: App startup and mode switching tested successfully

---

## Files Modified

1. `ui/apps/unified_ocr_app/components/inference/checkpoint_selector.py:186`
   - Changed widget key from `"mode_selector"` to `"inference_processing_mode_selector"`

2. `docs/CHANGELOG.md`
   - Added bug fix entry under "Fixed - 2025-10-21"

3. `docs/bug_reports/BUG-2025-012_streamlit_duplicate_element_key.md`
   - Created this comprehensive bug report

---

## Lessons Learned

1. **Global Key Scope**: Streamlit widget keys are globally scoped, not component-scoped
2. **Naming Convention**: Establish and follow consistent key naming patterns from the start
3. **Early Testing**: Test mode switching during development to catch key conflicts early
4. **Code Review**: Check for duplicate keys during code review process

---

**Resolution Confidence**: 100% - Root cause identified, fix verified, prevention guidelines established.
