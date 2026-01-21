

Output Log: Persistent Hydra Debugging Failure (Length: 10,765 CHARS)
```bash
*upstageailab-ocr-recsys-competition-ocr-2 ❯ bash -c 'if [[ "$VIRTUAL_ENV" == *".venv"* ]]; then echo "venv active: $VIRTUAL_ENV"; else echo "venv not active"; fi'
venv active: /workspaces/upstageailab-ocr-recsys-competition-ocr-2/.venv
*upstageailab-ocr-recsys-competition-ocr-2 ❯ grep -r "from ocr.domains.recognition" ocr/domains/detection/ 2>/dev/null || echo "✓ No detection→recognition imports"
✓ No detection→recognition imports
*upstageailab-ocr-recsys-competition-ocr-2 ❯ grep -r "from ocr.domains.detection" ocr/domains/recognition/ 2>/dev/null || echo "✓ No recognition→detection imports"
✓ No recognition→detection imports
*upstageailab-ocr-recsys-competition-ocr-2 ❯ grep -r "from ocr.domains" ocr/core/ | grep -E "(detection|recognition)" | head -20 || echo "No domain imports in core"
ocr/core/evaluation/__init__.py:from ocr.domains.detection.evaluation import CLEvalEvaluator
ocr/core/lightning/callbacks/wandb_image_logging.py:from ocr.domains.detection.callbacks.wandb import log_validation_images
ocr/core/lightning/ocr_pl.py.deprecated:from ocr.domains.detection.evaluation import CLEvalEvaluator
ocr/core/lightning/ocr_pl.py.deprecated:                        from ocr.domains.recognition.callbacks.wandb_logging import log_recognition_images
ocr/core/lightning/__init__.py:        from ocr.domains.detection.module import DetectionPLModule
ocr/core/lightning/__init__.py:        from ocr.domains.recognition.module import RecognitionPLModule
ocr/core/metrics/cleval_metric.py:from ocr.domains.detection.metrics.box_types import POLY
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/audit/arch_guard.py
✅ Architecture is compliant.
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py train domain=detection --dry-run
Error composing config: LexerNoViableAltException: --dry-run
                           ^
See https://hydra.cc/docs/1.2/advanced/override_grammar/basic for details
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py train domain=recognition --dry-run
Error composing config: LexerNoViableAltException: --dry-run
                           ^
See https://hydra.cc/docs/1.2/advanced/override_grammar/basic for details
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py train domain=detection 2>&1 | head -50
Error composing config: Cannot find primary config 'train'. Check that it's in your config search path.

Config search path:
        provider=hydra, path=pkg://hydra.conf
        provider=main, path=file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs
        provider=schema, path=structured://
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -100
Error composing config: In 'domain/detection': Could not override 'model/architectures@_group_.model.architectures'. No match in the defaults list.
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -150
Error composing config: In 'data/canonical': Could not find 'data/performance_preset/none'

Config search path:
        provider=hydra, path=pkg://hydra.conf
        provider=main, path=file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs
        provider=schema, path=structured://
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -100
Error composing config: In 'model/presets/dbnetpp': Could not find 'model/architectures/dbnetpp'

Config search path:
        provider=hydra, path=pkg://hydra.conf
        provider=main, path=file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs
        provider=schema, path=structured://
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -50
Error composing config: Interpolation key 'dataset_path' not found
    full_key: _group_.transforms.train_transform._target_
    object_type=dict
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -80
Error composing config: Interpolation key 'default_interpolation' not found
    full_key: _group_.transforms.train_transform.transforms[0].interpolation
    object_type=dict
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | grep -A 5 "default_interpolation"
Error composing config: Interpolation key 'default_interpolation' not found
    full_key: _group_.transforms.train_transform.transforms[0].interpolation
    object_type=dict
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -100
Error composing config: Interpolation key 'dataset_base_path' not found
    full_key: _group_.datasets.train_dataset.config.image_path
    object_type=dict
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -30
Error composing config: Interpolation key 'dataset_base_path' not found
    full_key: data.datasets.train_dataset.config.image_path
    object_type=dict
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -50
Error composing config: Interpolation key 'dataset_base_path' not found
    full_key: data.datasets.train_dataset.config.image_path
    object_type=dict
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -20
Error composing config: Interpolation key 'data.dataset_base_path' not found
    full_key: data.datasets.train_dataset.config.image_path
    object_type=dict
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -20
Error composing config: Interpolation key 'default_interpolation' not found
    full_key: transforms.transforms.train_transform.transforms[0].interpolation
    object_type=dict
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -50
Error composing config: Interpolation key 'default_interpolation' not found
    full_key: _group_.transforms.train_transform.transforms[0].interpolation
    object_type=dict
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -30
Error composing config: Interpolation key 'transforms.default_interpolation' not found
    full_key: _group_.transforms.train_transform.transforms[0].interpolation
    object_type=dict
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=recognition 2>&1 | grep -A 5 "transforms:" | head -20
  transforms:
    _target_: torchvision.transforms.Compose
    transforms:
    - _target_: torchvision.transforms.Resize
      size:
      - 224
      - 224
    - _target_: torchvision.transforms.ToTensor
--
      transforms:
      - _target_: torchvision.transforms.Resize
        size:
        - 224
        - 224
      - _target_: torchvision.transforms.ToTensor
--
        transforms:
        - _target_: torchvision.transforms.Resize
          size:
          - 224
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -30
Error composing config: Interpolation key 'dataset_base_path' not found
    full_key: _group_.datasets.train_dataset.config.image_path
    object_type=dict
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -10
Error composing config: Interpolation key 'data.dataset_base_path' not found
    full_key: _group_.datasets.train_dataset.config.image_path
    object_type=dict
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -5
Error composing config: Interpolation key 'transforms.train_transform' not found
    full_key: _group_.datasets.train_dataset.transform
    object_type=dict
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -5
Error composing config: Interpolation key 'train_transform' not found
    full_key: _group_.datasets.train_dataset.transform
    object_type=dict
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=recognition 2>&1 | grep -B 2 -A 10 "datasets:" | head -30
    logs: ./outputs/logs
    wandb: ./outputs/wandb
    datasets:
      train: ./data/train
      val: ./data/val
      test: ./data/test
    cache: ./outputs/cache
    temp: ./outputs/temp
dataset_path: ocr.data.datasets
dataset_config_path: ocr.core.validation
encoder_path: ocr.core.models.encoder
decoder_path: ocr.domains.detection.models.decoder
head_path: ocr.domains.detection.models.head
--
        - 0.225
    config: null
  datasets:
    train_dataset:
      _target_: ocr.features.recognition.data.lmdb_dataset.LMDBRecognitionDataset
      lmdb_path: ./data/train/recognition/aihub_lmdb_validation
      tokenizer:
        _target_: ocr.features.recognition.data.tokenizer.KoreanOCRTokenizer
        charset_path: ./ocr/data/charset.json
        max_len: 25
      max_len: 25
      transform:
        _target_: torchvision.transforms.Compose
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -2
Error composing config: Interpolation key 'train_transform' not found
    full_key: _group_.datasets.train_dataset.transform
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -1
Error composing config: Interpolation key 'transforms.train_transform' not found
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -50
Error composing config: Interpolation key 'train_transform' not found
    full_key: _group_.datasets.train_dataset.transform
    object_type=dict
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | grep -C 5 "train_transform"
    task_type: detection
    train_num_samples: null
    val_num_samples: null
    test_num_samples: null
  default_interpolation: 1
  train_transform:
    _target_: ocr.data.datasets.DBTransforms
    transforms:
    - _target_: albumentations.LongestMaxSize
      max_size: 640
      interpolation: 1


*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection 2>&1 | head -50
Error composing config: Interpolation key 'data.train_transform' not found
    full_key: data.datasets.train_dataset.transform
    object_type=dict
```

## Additional errors:

1. Nonexistent key: 'model/architectures@_group_.model.architectures'
```bash
*upstageailab-ocr-recsys-competition-ocr-2 ❯ python scripts/utils/show_config.py main domain=detection
Error composing config: In 'domain/detection': Could not override 'model/architectures@_group_.model.architectures'. No match in the defaults list.
```


2. ModuleNotFoundError: No module named 'ocr.pipelines.config_loader'
```bash
*upstageailab-ocr-recsys-competition-ocr-2 ❯ uv run python -m runners.train --info defaults-tree
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/runners/train.py", line 5, in <module>
    from ocr.pipelines.orchestrator import OCRProjectOrchestrator
  File "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/pipelines/orchestrator.py", line 23, in <module>
    from .config_loader import PostprocessSettings
ModuleNotFoundError: No module named 'ocr.pipelines.config_loader'
```
