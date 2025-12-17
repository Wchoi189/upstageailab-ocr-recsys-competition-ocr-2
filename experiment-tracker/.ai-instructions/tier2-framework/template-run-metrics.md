---
ads_version: "1.0"
type: "report"
experiment_id: "[EXPERIMENT_ID]"
status: "active"
created: "[TIMESTAMP]"
updated: "[TIMESTAMP]"
tags: ["metrics", "[PHASE_TAG]"]
metrics: ["metric1", "metric2", "metric3"]
baseline: "run_001"
comparison: "baseline"
---

# Run Metrics: [Phase Name]

**Phase**: [phase_name]
**Metric Focus**: [Primary metrics being tracked]
**Baseline**: [Reference run for comparison]

## Run History

| Run | Date | Parameters | Metric1 | Metric2 | Metric3 | Status | Notes |
|-----|------|------------|---------|---------|---------|--------|-------|
| 001 | YYYY-MM-DD | [params] | X.XX | XXms | XX% | ✅ | Baseline |

## Best Performance

| Metric | Best Value | Run | Date | Parameters |
|--------|------------|-----|------|------------|
| Metric1 | X.XX | 001 | YYYY-MM-DD | [params] |
| Metric2 | XXms | 001 | YYYY-MM-DD | [params] |
| Metric3 | XX% | 001 | YYYY-MM-DD | [params] |

## Trend Analysis

- **Metric1**: [Trend description]
- **Metric2**: [Trend description]
- **Metric3**: [Trend description]

## Current Recommendation

**Deploy**: Run [XXX] ([reason])

---

## Detailed Notes

### Run 001: Baseline
[Detailed description of baseline configuration and results]

### Run 002+: Experiments
[Add detailed notes for each subsequent run as needed]
