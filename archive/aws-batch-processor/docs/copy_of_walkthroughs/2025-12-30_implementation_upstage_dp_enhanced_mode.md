# Goal: Enable Enhanced Mode in Document Parse Processing

Update [scripts/process_document_parse.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/scripts/process_document_parse.py) to support the user's request for "enhanced mode". This improves accuracy for complex elements (tables, handwriting).

## User Review Required
> [!IMPORTANT]
> **Model Change**: Enhanced mode works best with or requires `model="document-parse-nightly"` according to documentation. I will update the script to default to this when `--enhanced` is used, but allow override.

## Proposed Changes

### [scripts/process_document_parse.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/scripts/process_document_parse.py)
#### [MODIFY] Update [process_image](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/scripts/process_document_parse.py#30-100) and [main](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/scripts/process_document_parse.py#101-255)
- Add `--enhanced` argument to [main](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/scripts/process_document_parse.py#101-255).
- Update [process_image](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/scripts/process_document_parse.py#30-100) to accept `enhanced` boolean.
- In [process_image](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/scripts/process_document_parse.py#30-100), if `enhanced` is True:
    - Set `data["mode"] = "enhanced"`
    - Set `data["model"] = "document-parse-nightly"` (unless implicitly handled, but explicit is safer based on docs).

## Verification Plan

### Automated Tests
- Run [test_api_capabilities.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/scripts/test_api_capabilities.py) (modified) or a new test command using the updated script on [test_poly.jpg](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/test_poly.jpg) with `--enhanced` flag.
- Verify output contains `mode: enhanced` metadata or successful processing.

```bash
python3 scripts/process_document_parse.py --dataset=test_local_enhanced --local-only --resume --enhanced
```
