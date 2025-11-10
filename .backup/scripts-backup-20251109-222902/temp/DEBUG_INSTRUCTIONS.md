# Debug Instructions for Streamlit Freeze

You're right - I've been making hypotheses without evidence. Let's get actual data.

## Step 1: Run Minimal Test (No Streamlit)

This tests if the inference engine itself works:

```bash
uv run python test_inference_minimal.py
```

**Expected**: Should complete all 8 tests successfully
**If it fails**: Note exactly which test number fails
**If it succeeds**: Problem is Streamlit-specific, not inference engine

## Step 2: Enable Debug Logging in Streamlit

Add this line to the **very top** of `ui/apps/inference/app.py`:

```python
from .debug_logging import LOGGER, log_checkpoint, log_memory_usage, trace_call
```

Then add checkpoints throughout the code:

```python
# At the start of run() function
log_checkpoint("START: app.run()")
log_memory_usage()

# After loading catalog
log_checkpoint("Catalog loaded")

# After sidebar renders
log_checkpoint("Sidebar rendered")

# Before inference
log_checkpoint("BEFORE: inference_service.run()")

# After inference
log_checkpoint("AFTER: inference_service.run()")

# Before results display
log_checkpoint("BEFORE: results_component.render_results()")

# After results display
log_checkpoint("AFTER: results_component.render_results()")
```

## Step 3: Run Streamlit and Trigger Freeze

```bash
make stop-inference-ui
make ui-infer
```

Then:
1. Upload ONE image
2. Click "Run Inference"
3. **Watch the terminal for the last log message before freeze**

## Step 4: Check the Log File

```bash
tail -100 /tmp/streamlit_debug.log
```

**The last log message tells us exactly where it freezes.**

## What to Look For

### If last message is "BEFORE: inference_service.run()":
→ Problem is in the inference engine itself

### If last message is "AFTER: inference_service.run()":
→ Problem is between inference and display

### If last message is "BEFORE: results_component.render_results()":
→ Problem is in results rendering

### If last message is "BEFORE: st.image(...)":
→ Problem is Streamlit's image display

### If no logs appear at all:
→ Problem is before app even runs (configuration/imports)

## Step 5: Report Back

**Please run these tests and tell me:**

1. Does `test_inference_minimal.py` succeed? (YES/NO)
2. What is the **last log message** before freeze in `/tmp/streamlit_debug.log`?
3. How long does it take to freeze? (Immediate? After 30s? After 60s?)
4. Does it freeze on the FIRST inference or only after multiple?

**With this data, I can pinpoint the exact line causing the freeze.**

---

## Alternative: System Resource Monitoring

While Streamlit is running, open another terminal and run:

```bash
# Monitor CPU and memory every 2 seconds
watch -n 2 'ps aux | grep streamlit | grep -v grep; free -h'
```

**Look for:**
- CPU at 100% → Infinite loop or heavy computation
- Memory growing → Memory leak
- No activity → App is truly frozen/deadlocked

---

## Quick Test: Disable Image Display

As a quick test, comment out the image display:

In `ui/apps/inference/components/results.py`, line 182:

```python
if result.image is not None and predictions:
    # _display_image_with_predictions(result.image, predictions, config)
    st.warning("⚠️ Image display temporarily disabled for testing")
```

Then run inference again. If it works → problem is in image display. If still freezes → problem is earlier.

---

## Evidence-Based Approach

I apologize for guessing. Let's get real data:

✅ **Test 1**: Minimal script (isolates inference engine)
✅ **Test 2**: Debug logging (finds exact freeze point)
✅ **Test 3**: Resource monitoring (identifies bottleneck type)
✅ **Test 4**: Selective disabling (narrows scope)

**Please run these and share the results. Then I can fix the actual root cause, not a hypothesis.**
