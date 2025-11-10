---
title: "Assessment: GPU Stability vs PSU Headroom"
date: "2025-11-10"
type: "assessment"
category: "hardware"
status: "open"
version: "1.0"
tags:
  - power-supply
  - gpu-stability
  - bug-20251110-002
---

## Summary
Training crashes (`BUG-20251110-002`) align with symptoms of an undersized power supply. The RTX 3060 system currently runs ~50 W below NVIDIA’s recommended PSU capacity, leaving little headroom. Under load, transient spikes in GPU draw can exceed what the PSU can deliver, causing voltage droop and driver faults (`CUDNN_STATUS_EXECUTION_FAILED`, illegal instruction). Upgrading the PSU mitigates the most plausible hardware root cause and reduces risk of future GPU instability.

## Evidence
- Crash occurs during high-load convolutions (FPN decoder), consistent with peak draw periods.
- Stack trace shows cuDNN execution failure followed by device teardown errors, typical of GPU power/instability events.
- NVIDIA guidance for RTX 3060 is ≥550 W PSU; local system is ~50 W below that, offering <10 % headroom.
- Similar cases (NVIDIA forums, PyTorch issues) report identical error signatures resolved by PSU upgrade.

## Options
1. **Upgrade PSU (recommended)**
   - Install 650–750 W 80+ Gold (or better) unit.
   - Pros: ample 12 V rail current, accommodates future GPU/CPU upgrades, lowest risk.
   - Cons: upfront cost (~$80–$130), installation time.

2. **Apply power caps / software mitigations**
   - Set GPU power limit (e.g., `nvidia-smi -pl 160`) or reduce batch size.
   - Pros: zero cost quick test.
   - Cons: lower performance, still vulnerable to spikes; not a long-term fix.

3. **Do nothing**
   - Pros: no cost.
   - Cons: High probability of recurring crashes, potential data corruption, long debug cycles.

## Recommendation
Proceed with Option 1. A modern 650–750 W PSU provides enough sustained and transient power for the RTX 3060 + CPU combo, eliminating a key instability source. Keep Option 2 as a stopgap until hardware arrives, but avoid production runs until power delivery is upgraded.

## Next Steps
- Select reputable PSU (e.g., Seasonic Focus, Corsair RM series) rated ≥650 W with sufficient PCIe connectors.
- Install and retest the training run to confirm stability.
- Update `BUG-20251110-002` with results after hardware change.


