# 🔧 Streamlit Freeze Debugging Toolkit

**Problem:** Streamlit inference app freezes after inference completes, when displaying results. No error messages.

**Evidence so far:**
- ✅ Inference engine works perfectly in isolation
- ✅ Predictions complete successfully (shows avg confidence)
- ❌ Freezes during/after results display
- ❌ No error messages in logs
- ❌ Print statements not appearing in log files

---

## 🎯 Recommended Approach (Try in Order)

### 1. Run in Foreground Mode ⭐ **START HERE**

**Why:** Direct terminal output bypasses log file issues. You'll see exactly where it freezes.

```bash
./test_freeze_scenarios.sh
# Choose option 1
```

Or manually:
```bash
make stop-inference-ui
cd ui/apps/inference
uv run streamlit run app.py --server.port=8504
```

**What to watch:** The terminal will show print statements in real-time. Note the LAST message before freeze.

**Expected output:**
```
>>> APP.RUN() STARTED
>>> CALLING render_results
>>> render_results CALLED
    Header rendered
    Calling _render_results_table
        Calling st.dataframe          ← If freezes here: st.dataframe issue
        st.dataframe completed
    Creating expander for result 1   ← If freezes here: st.expander issue
```

---

### 2. Test Widgets in Isolation

**Why:** Identifies which specific widget causes the freeze.

```bash
uv run streamlit run test_streamlit_widgets.py
```

Then click each test button:
- Test 1: st.dataframe() alone
- Test 2: st.expander() alone
- Test 3: st.image() alone
- Test 4: Combined

**If a test freezes:** That widget is the culprit.

**If all tests pass:** Problem is with the data being passed, not the widgets themselves.

---

### 3. Use Minimal Renderer

**Why:** Completely disables st.dataframe() and st.expander() to test if they're the issue.

```bash
./test_freeze_scenarios.sh
# Choose option 2 (switches to minimal renderer)
# Then choose option 1 or 3 to run the app
```

Or manually:
```bash
# Backup original
cp ui/apps/inference/components/results.py ui/apps/inference/components/results.py.BACKUP

# Use minimal version (text-only display)
cp ui/apps/inference/components/results_minimal.py ui/apps/inference/components/results.py

# Run app
make stop-inference-ui
make ui-infer
```

**If minimal version works:** Problem is definitely in st.dataframe() or st.expander().

**If minimal version still freezes:** Problem is earlier in the pipeline (inference_runner or state management).

---

### 4. Check Logs After Freeze

```bash
bash CHECK_LOGS.sh
```

Shows the last 50 lines of both output and error logs.

---

## 📂 Files in This Toolkit

### Diagnostic Scripts
- **`test_freeze_scenarios.sh`** - Interactive menu for different test modes
- **`test_streamlit_widgets.py`** - Test each widget independently
- **`test_inference_minimal.py`** - Test inference without Streamlit
- **`CHECK_LOGS.sh`** - Quick log checker

### Alternative Implementations
- **`ui/apps/inference/components/results_minimal.py`** - Text-only results display (no widgets)

### Documentation
- **`FREEZE_DEBUG_PLAN.md`** - Detailed step-by-step plan
- **`DEBUG_INSTRUCTIONS.md`** - Evidence-based diagnostic approach
- **`WATCH_OUTPUT.md`** - How to monitor logs in real-time
- **`INFERENCE_CRASH_SUSPECTS.md`** - List of potential culprits

### Debug Utilities
- **`ui/apps/inference/debug_logging.py`** - Logging utilities (already imported in app)

---

## 🔍 What Each Test Tells Us

| Test Result | Conclusion | Next Step |
|-------------|------------|-----------|
| Foreground shows freeze at "Calling st.dataframe" | st.dataframe() is the problem | Replace with st.table() or simple markdown |
| Foreground shows freeze at "Creating expander" | st.expander() is the problem | Use st.columns() or disable detailed view |
| Widget test #1 freezes | st.dataframe() confirmed | Check data size/format |
| Widget test #2 freezes | st.expander() confirmed | Limit number of expanders |
| Minimal renderer works | Widgets are the problem | Use alternative widgets |
| Minimal renderer freezes | Problem is before display | Check inference_runner.py |
| All tests pass, only real app freezes | Data-specific issue | Check what data is being passed |

---

## 🎓 Understanding the Code Flow

```
app.py:run()
    ↓
sidebar_component.render_controls()
    ↓ (user clicks "Run Inference")
inference_service.run()
    ↓
InferenceEngine.predict()  ← This works (tested independently)
    ↓
state.inference_results.append(...)
    ↓
results_component.render_results()
    ↓
    ├─ st.header("Results")
    ├─ _render_results_table()
    │   └─ st.dataframe(df)  ← LIKELY FREEZE POINT
    └─ for result in results:
        └─ st.expander(...)  ← ALSO POSSIBLE FREEZE POINT
            └─ st.image(...)
```

---

## 🚨 Known Issues

1. **Print statements not showing in logs**
   - Log files might be buffered
   - Streamlit might capture stderr
   - Solution: Run in foreground mode

2. **Session state accumulation**
   - Multiple runs add to state.inference_results
   - Large arrays in memory
   - Solution: Limited to MAX_RESULTS_IN_MEMORY = 10 (already implemented)

3. **Threading timeout removed**
   - Previous signal-based timeout caused crashes
   - Thread-based timeout also caused issues
   - Now using direct inference call

---

## 💡 Quick Fixes to Try

### Fix A: Replace st.dataframe with st.table

In `ui/apps/inference/components/results.py` line 133:
```python
# BEFORE
st.dataframe(df, use_container_width=True)

# AFTER
st.table(df)  # Less feature-rich but more stable
```

### Fix B: Limit expanders

In `ui/apps/inference/components/results.py` line 144:
```python
# BEFORE
for idx, result in enumerate(state.inference_results, 1):

# AFTER
for idx, result in enumerate(state.inference_results[-3:], 1):  # Only last 3
```

### Fix C: Lazy load expanders (only expand one at a time)

```python
# Use st.selectbox instead of multiple expanders
selected_idx = st.selectbox("Select result to view", range(1, len(state.inference_results) + 1))
result = state.inference_results[selected_idx - 1]
# Then display just that one result
```

---

## 📞 What Information Would Help

If you report back, please include:

1. **Which test did you run?** (Foreground? Widget test? Minimal renderer?)
2. **Last message printed** before freeze (from terminal if foreground mode)
3. **Which test number** caused freeze (if using widget test)
4. **Does minimal renderer work?** (Yes/No)
5. **How many images** did you test with? (1? Multiple?)

With this information, I can provide a precise fix.

---

## 🎯 Most Likely Culprits (Ranked)

Based on "shows avg confidence then freezes":

1. **st.dataframe() with large data** - 70%
   - Summary renders (has avg confidence)
   - Dataframe comes right after
   - CPU-intensive with many rows

2. **st.expander() in loop** - 20%
   - Creating many expanders at once
   - Streamlit pre-rendering issue

3. **Session state corruption** - 10%
   - Multiple reruns accumulating data
   - Already limited to 10 results

---

**Remember:** We're done guessing. These tools give us EVIDENCE. Please try them and share what you find! 🔬
