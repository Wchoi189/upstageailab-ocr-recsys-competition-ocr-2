# Streamlit Inference App Crash Suspects

## Critical Issue
App crashes after predictions are made, specifically when displaying results.

---

## 🔴 HIGH PRIORITY SUSPECTS

### 1. **Threading Timeout in engine.py** ⚠️ CRITICAL
**File**: `ui/utils/inference/engine.py`
**Location**: `_run_with_timeout()` function (lines 37-77)
**Issue**: Thread-based timeout creates daemon threads that may not properly cleanup
**Symptoms**:
- Hanging threads consuming resources
- Memory not released after inference
- App freezes after multiple inferences

**Potential Problem**:
```python
thread = threading.Thread(target=_wrapper)
thread.daemon = True  # ⚠️ Daemon threads can cause issues
thread.start()
thread.join(timeout=timeout_seconds)

if thread.is_alive():  # ⚠️ Thread still running!
    # We raise TimeoutError but thread keeps running
    raise TimeoutError(...)
```

**Why it crashes**:
- Daemon thread continues running even after timeout
- Thread holds references to model/tensors in GPU memory
- Multiple hanging threads accumulate
- Eventually causes OOM or app freeze

**Test**: Comment out the timeout wrapper and see if crash persists

---

### 2. **Model Loading in Thread Context**
**File**: `ui/utils/inference/engine.py`
**Location**: `InferenceEngine.load_model()` and `predict_image()`
**Issue**: PyTorch/CUDA operations in non-main thread
**Symptoms**:
- Works for first inference, crashes on subsequent ones
- GPU memory not released
- CUDA context errors

**Potential Problem**:
```python
# Model loaded in Streamlit thread, not main thread
self.model = model.to(self.device)

# Then used in timeout thread (nested threading!)
with torch.no_grad():
    predictions = self.model(return_loss=False, images=batch.to(self.device))
```

**Why it crashes**:
- CUDA contexts don't always work well with threading
- GPU memory leaks between inferences
- Nested threading (Streamlit thread → timeout thread) compounds issues

---

### 3. **Image Array Memory Leak**
**File**: `ui/apps/inference/components/results.py`
**Location**: `_display_image_with_predictions()` (lines 193-261)
**Issue**: Large image arrays kept in memory
**Symptoms**:
- Memory grows with each inference
- App slows down over time
- Eventually crashes

**Potential Problem**:
```python
# Image array stored in state.inference_results
state.inference_results.append(result)  # ⚠️ Accumulates in memory

# Result contains full-size image array
result = InferenceResult(
    image=inference_rgb,  # ⚠️ Large numpy array in session state
    predictions=predictions,
    ...
)
```

**Why it crashes**:
- Streamlit session state keeps ALL results in memory
- Each image is ~10-50MB uncompressed
- 10 images = 100-500MB in session state
- Eventually exceeds memory limits

---

### 4. **PIL Image Draw Operations**
**File**: `ui/apps/inference/components/results.py`
**Location**: Drawing polygons with PIL (lines 215-223)
**Issue**: Too many polygons or complex drawings
**Symptoms**:
- Crashes specifically during display
- Works with few detections, crashes with many

**Potential Problem**:
```python
for index, polygon_str in enumerate(polygons):
    points = _parse_polygon_points(polygon_str)
    draw.polygon(points, outline=(255, 0, 0, 255), fill=(255, 0, 0, 30))
    # Drawing 100+ polygons with transparency can be slow/crash
```

**Why it crashes**:
- PIL ImageDraw with RGBA transparency is slow
- Many polygons (85+) * transparency layers = CPU intensive
- Can freeze/timeout Streamlit connection

---

## 🟡 MEDIUM PRIORITY SUSPECTS

### 5. **Preprocessing Image Storage**
**File**: `ui/apps/inference/services/inference_runner.py`
**Location**: Lines 146-172 (preprocessing section)
**Issue**: Stores both original AND processed images
**Symptoms**: Memory grows with preprocessing enabled

### 6. **Checkpoint Loading**
**File**: `ui/utils/inference/model_loader.py`
**Issue**: Model not properly released between inferences
**Test**: Check if crashes happen with same checkpoint or when switching

### 7. **Config Caching**
**File**: `ui/apps/inference/app.py`
**Location**: `@st.cache_data` decorators
**Issue**: Cached objects holding memory

---

## 🟢 LOW PRIORITY SUSPECTS

### 8. **Pandas DataFrame in Results**
**File**: `ui/apps/inference/components/results.py`
**Location**: Line 164 - `st.dataframe(df)`

### 9. **JSON Dumps in Expanders**
**File**: `ui/apps/inference/components/results.py`
**Location**: Line 190 - `st.json(predictions.model_dump())`

---

## 🔍 DIAGNOSTIC TESTS

### Test 1: Isolate Threading Issue
```python
# In ui/utils/inference/engine.py
# Replace predict_image() to NOT use timeout:

# Comment out this line:
# predictions = _run_with_timeout(_inference_func, timeout_seconds=60)

# Replace with direct call:
predictions = _inference_func()
```

If crash stops → Threading timeout is the culprit

---

### Test 2: Disable Image Storage in State
```python
# In ui/apps/inference/services/inference_runner.py
# Line 209-228, replace with:

result = InferenceResult(
    filename=filename,
    success=True,
    image=None,  # ⚠️ Don't store image
    predictions=predictions,
    preprocessing=preprocessing_info,
)
```

If crash stops → Memory accumulation is the culprit

---

### Test 3: Disable Drawing Polygons
```python
# In ui/apps/inference/components/results.py
# Line 217-223, comment out the drawing loop:

if predictions.polygons:
    pass  # ⚠️ Skip drawing
    # polygons = predictions.polygons.split("|")
    # ... drawing code ...
```

If crash stops → PIL drawing is the culprit

---

### Test 4: Limit Results History
```python
# In ui/apps/inference/services/inference_runner.py
# After line 120:

state.inference_results.extend(new_results)

# Add this:
if len(state.inference_results) > 5:
    state.inference_results = state.inference_results[-5:]  # Keep only last 5
```

If crash is delayed → Memory accumulation is the culprit

---

## 📊 LIKELIHOOD RANKING

Based on symptoms (crash after display, connection timeout):

1. **#1 Threading Timeout** - 🔴 **90% LIKELY**
   - Daemon threads not cleaning up properly
   - Nested threading with Streamlit + timeout wrapper

2. **#3 Image Memory Leak** - 🔴 **80% LIKELY**
   - Session state accumulating large arrays
   - Multiple inferences exhaust memory

3. **#4 PIL Drawing** - 🟡 **60% LIKELY**
   - CPU-intensive with many polygons
   - Can cause connection timeout

4. **#2 Model Threading** - 🟡 **50% LIKELY**
   - CUDA context issues
   - GPU memory not released

5. **Others** - 🟢 **<30% LIKELY**

---

## 🔧 RECOMMENDED ACTION PLAN

### Step 1: Quick Win - Remove Timeout Wrapper
The threading timeout is most likely causing issues. Streamlit has its own timeout mechanism.

**Change**: Remove `_run_with_timeout()` usage:
```python
# In ui/utils/inference/engine.py, line 245:
# predictions = _run_with_timeout(_inference_func, timeout_seconds=60)

# Replace with:
predictions = _inference_func()
```

### Step 2: Limit Session State Size
**Change**: Don't store images in session state, or limit history:
```python
# In ui/apps/inference/services/inference_runner.py
# Keep only last 3 results
state.inference_results = state.inference_results[-3:]
```

### Step 3: Optimize Drawing
**Change**: Skip drawing if too many polygons:
```python
# In results.py
if len(polygons) > 100:
    st.warning("⚠️ Too many detections to display (>100). Showing count only.")
    return
```

---

## 🚨 IMMEDIATE TEST

Run this minimal test to isolate the issue:

```bash
# Kill existing process
make stop-inference-ui

# Edit engine.py to remove timeout
# Edit results.py to limit results to 3

# Restart
make ui-infer

# Test with ONE image
# If works → Try TWO images
# If crashes → Issue is accumulation
# If still crashes on first → Issue is threading/GPU
```

---

## 📝 FILES TO CHECK/MODIFY

**Priority 1 (Check First)**:
1. `ui/utils/inference/engine.py` - Lines 37-77, 245
2. `ui/apps/inference/services/inference_runner.py` - Lines 120-125, 209-228
3. `ui/apps/inference/components/results.py` - Lines 193-261

**Priority 2 (Check If Above Don't Help)**:
4. `ui/utils/inference/model_loader.py`
5. `ui/apps/inference/state.py`
6. `ui/apps/inference/app.py`

---

**Created**: 2025-10-19
**Status**: Diagnostic document for active crash investigation
