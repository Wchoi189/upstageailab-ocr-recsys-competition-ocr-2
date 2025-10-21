# Freeze Debug Plan - Direct Approach

## Current Status
- Print statements to stderr not appearing in logs
- App freezes after inference completes
- Shows avg confidence, then freezes
- No error messages

## Next Steps (Evidence-Based)

### Step 1: Run in Foreground Mode ⭐ TRY THIS FIRST

This will show all output directly in the terminal, bypassing log file issues.

```bash
# Stop background process
make stop-inference-ui

# Run in foreground - you'll see all print statements in real-time
cd ui/apps/inference
uv run streamlit run app.py --server.port=8504
```

**What to watch for:**
- The terminal will show all print statements as they happen
- The LAST message printed before freeze tells us exactly where it stops
- You should see the `>>> render_results CALLED` messages

**Expected output:**
```
>>> APP.RUN() STARTED
>>> CALLING render_results
>>> render_results CALLED
    Header rendered
    Calling _render_results_table
        Calling st.dataframe          ← If it freezes here, problem is st.dataframe()
        st.dataframe completed
    _render_results_table completed
    Starting detailed results loop
    Processing result 1/X
    Creating expander for result 1   ← If it freezes here, problem is st.expander()
```

### Step 2: Binary Search Disable (If Step 1 doesn't work)

If we still can't see where it freezes, disable components one by one:

**Test A - Disable st.dataframe:**
In `ui/apps/inference/components/results.py`, comment out lines 127-133:
```python
# st.divider()
# st.markdown("### 📋 Summary")
# _render_results_table(state.inference_results)
st.markdown("**⚠️ Summary table disabled for debugging**")
st.markdown(f"Total results: {len(state.inference_results)}")
```

**If Test A works (no freeze)** → Problem is in `_render_results_table()` or `st.dataframe()`

**Test B - Disable detailed results:**
If Test A still freezes, comment out lines 135-174:
```python
# st.divider()
# st.markdown("### 🔍 Detailed Results")
# for idx, result in enumerate(state.inference_results, 1):
#     ... entire loop ...
st.markdown("**⚠️ Detailed results disabled for debugging**")
```

**If Test B works (no freeze)** → Problem is in the expander loop

### Step 3: Check Streamlit Session State

The freeze might be caused by corrupted session state. Add this at the START of `render_results()`:

```python
def render_results(state: InferenceState, config: UIConfig) -> None:
    import sys

    # Check session state health
    print(f"DEBUG: Number of results: {len(state.inference_results)}", file=sys.stderr, flush=True)
    print(f"DEBUG: Session state keys: {st.session_state.keys()}", file=sys.stderr, flush=True)

    # Clear and restart if state is corrupted
    if len(state.inference_results) > 20:
        print("WARNING: Too many results in state, clearing...", file=sys.stderr, flush=True)
        state.inference_results = state.inference_results[-5:]
        state.persist()
```

### Step 4: Nuclear Option - Minimal Results Display

Replace the entire `render_results()` function with this minimal version:

```python
def render_results(state: InferenceState, config: UIConfig) -> None:
    st.header("📊 Inference Results")

    if not state.inference_results:
        st.info("No results yet. Run inference to see predictions.")
        return

    # Only show text, no widgets
    st.write(f"Total results: {len(state.inference_results)}")

    for idx, result in enumerate(state.inference_results[-3:], 1):  # Only last 3
        st.write(f"Result {idx}: {result.num_predictions} predictions")
```

**If this works** → Problem is definitely in the widget rendering (st.dataframe/st.expander)

---

## Most Likely Root Causes (Based on Evidence)

1. **st.dataframe() with large data** (70% likely)
   - Symptom: Freezes right after showing avg confidence
   - Avg confidence is rendered BEFORE the dataframe
   - Dataframe creation might be CPU-intensive

2. **st.expander() with many items** (20% likely)
   - Creating expanders in a loop might cause issues
   - Streamlit might be trying to pre-render all expanders

3. **Session state corruption** (10% likely)
   - Multiple reruns accumulating state
   - Need to limit results in memory

---

## Action Plan

1. **Try Step 1 (Foreground mode)** - Most direct way to see what's happening
2. **If still unclear, try Step 4 (Minimal display)** - Proves it's a widget issue
3. **Then binary search (Step 2)** - Isolate which widget
4. **Then fix the specific widget** - Replace with alternative or optimize

**Please start with Step 1 (foreground mode) and share what you see in the terminal!**
