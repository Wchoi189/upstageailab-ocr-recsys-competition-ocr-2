# Page snapshot

```yaml
- generic [ref=e3]:
  - heading "Comparison Studio" [level=1] [ref=e4]
  - paragraph [ref=e5]: Configure parameter sweeps and compare model performance
  - generic [ref=e6]:
    - generic [ref=e7]:
      - generic [ref=e8]:
        - heading "Select Preset" [level=3] [ref=e9]
        - combobox [ref=e10]:
          - option "Single Run Analysis" [selected]
          - option "Model A/B Comparison"
          - option "Image Gallery"
        - paragraph [ref=e11]: Metrics + gallery for a single submission file.
      - generic [ref=e12]:
        - heading "Parameters" [level=3] [ref=e13]
        - generic [ref=e15]:
          - generic [ref=e16]: Model A Path *
          - textbox "Model A Path *" [ref=e17]:
            - /placeholder: outputs/exp_1/predictions.csv
        - generic [ref=e19]:
          - generic [ref=e20]: Ground Truth Path (optional)
          - textbox "Ground Truth Path (optional)" [ref=e21]:
            - /placeholder: data/labels/ground_truth.csv
        - button "Run Comparison" [ref=e22] [cursor=pointer]
    - generic [ref=e23]:
      - heading "Results" [level=3] [ref=e24]
      - paragraph [ref=e26]: Configure parameters and run comparison to see results
```