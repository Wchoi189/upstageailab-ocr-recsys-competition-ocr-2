# Page snapshot

```yaml
- generic [ref=e3]:
  - heading "Command Builder" [level=1] [ref=e4]
  - paragraph [ref=e5]: Build metadata-aware training, testing, and prediction commands
  - generic [ref=e6]:
    - button "Training" [ref=e7] [cursor=pointer]
    - button "Testing" [ref=e8] [cursor=pointer]
    - button "Prediction" [ref=e9] [cursor=pointer]
  - generic [ref=e10]:
    - heading "⚠️ Error Loading Schema" [level=3] [ref=e11]
    - paragraph [ref=e12]: Request timed out. The API server may not be running. Make sure it's running at http://127.0.0.1:8000
    - paragraph [ref=e13]:
      - strong [ref=e14]: "To fix this:"
      - text: "1. Make sure the API server is running:"
      - code [ref=e15]: uv run python run_spa.py --api-only
      - text: 2. Check that the server is accessible at
      - code [ref=e16]: http://127.0.0.1:8000
      - text: 3. Check the browser console (F12) for more details
```
