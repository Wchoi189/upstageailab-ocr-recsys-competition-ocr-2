# Component Protocols

**Purpose:** Concise instructions for component-specific tasks. For detailed context, see `docs/maintainers/protocols/components/`.

## Training Protocol

**Planning:**
- Create experiment plan (use QMF toolbelt)
- Define hypothesis and configuration
- Document in `docs/maintainers/experiments/`

**Execution:**
```bash
# Start context log
make context-log-start LABEL="experiment_name"

# Run training
uv run python runners/train.py preset=<name> [overrides]

# Monitor via W&B dashboard
```

**Analysis:**
- Record key metrics: val/hmean, test/recall, test/precision
- Link W&B run in experiment log
- Document findings and next steps

**Tools:**
```bash
# Collect results
uv run python scripts/collect_results.py

# Generate ablation table
uv run python scripts/generate_ablation_table.py
```

## Streamlit Protocols

**Coding Rules:**
- NEVER use `use_container_width` (deprecated)
- ALWAYS use `width="stretch"` or `width="content"`
- Applies to: `st.plotly_chart()`, `st.dataframe()`, `st.button()`

**Debugging:**
- Test in browser (not just unit tests)
- Check logs: `python scripts/process_monitor.py`
- Use `st.write()` for debugging output

**Refactoring:**
- Follow component-based architecture
- Use `run_ui.py` for app entry points
- Maintain backward compatibility

**Maintenance:**
- Update dependencies: `uv sync`
- Test all pages after changes
- Check for duplicate element keys

## Preprocessing Workflow

**docTR Preprocessing:**
- Use preprocessing profiles from configs
- Validate data contracts with Pydantic
- Test with sample data before full run

**Checkpoint Migration:**
- Use checkpoint catalog for discovery
- Validate checkpoint compatibility
- Update naming scheme if needed

## Template Adoption

**Process:**
1. Review template requirements
2. Check compatibility with existing code
3. Test with sample data
4. Update documentation

## Advanced Training Techniques

**Techniques:**
- Mixed precision: `trainer.precision=16-mixed`
- Gradient accumulation: `trainer.accumulate_grad_batches=N`
- Learning rate scheduling: Configure in model config
- Checkpoint callbacks: Use PyTorch Lightning callbacks

**Performance:**
- Enable image preloading and caching
- Use tensor caching for faster iterations
- Monitor GPU/CPU usage

