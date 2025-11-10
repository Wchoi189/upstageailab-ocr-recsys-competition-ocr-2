# Watch Streamlit Output in Real-Time

The print statements now go directly to stderr and should appear in the logs.

## Method 1: Check Error Log After Freeze

```bash
# After the app freezes, check the error log:
tail -100 logs/ui/inference_8504.err
```

Look for the **last message** before the freeze. It will be one of these:

```
>>> APP.RUN() STARTED
>>> CALLING render_results
>>> render_results CALLED
    Header rendered
    Calling _render_results_table
        Calling st.dataframe          ← LIKELY FREEZE POINT
        st.dataframe completed
    _render_results_table completed
    Starting detailed results loop
    Processing result 1/X
    Creating expander for result 1   ← ALSO POSSIBLE FREEZE POINT
<<< render_results COMPLETED
<<< APP.RUN() COMPLETED
```

## Method 2: Watch in Real-Time

Open **TWO terminals**:

### Terminal 1 - Run the app:
```bash
make stop-inference-ui
make ui-infer
```

### Terminal 2 - Watch output:
```bash
tail -f logs/ui/inference_8504.err
```

Then run inference in the browser and watch Terminal 2 for the last message.

## Method 3: Direct Terminal Output

If logs still don't show, run Streamlit directly in terminal to see output:

```bash
# Stop the background process
make stop-inference-ui

# Run directly in terminal (output visible)
cd ui/apps/inference
uv run streamlit run app.py --server.port=8504
```

Then watch the terminal output directly.

---

## What Each Message Means

**Last message → Where it freezes:**

| Last Message | Freeze Location | Likely Cause |
|--------------|-----------------|--------------|
| `>>> CALLING render_results` | **BEFORE** results display starts | Streamlit routing issue |
| `>>> render_results CALLED` | Start of render_results() | Session state issue |
| `Header rendered` | After st.header() | Batch output section |
| `Calling _render_results_table` | **BEFORE** table creation | Summary or divider |
| `Calling st.dataframe` | **AT** st.dataframe() call | **PANDAS/DATAFRAME BUG** ⚠️ |
| `st.dataframe completed` | After dataframe | Divider or expander setup |
| `Creating expander` | **AT** st.expander() call | **EXPANDER BUG** ⚠️ |
| `Calling _render_single_result` | Inside expander | Display logic |

## Most Likely Culprits

Based on "shows avg confidence then freezes":

1. **st.dataframe()** - 70% likely
2. **st.expander()** - 20% likely
3. **Something else** - 10% likely

---

**Please try one of these methods and share the last message you see!**
