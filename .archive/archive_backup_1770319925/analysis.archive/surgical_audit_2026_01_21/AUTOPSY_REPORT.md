# ocr/core Surgical Autopsy Report

| File | LOC | Illegal Imports | Domain Keywords |
|---|---|---|---|
| `lightning/callbacks/wandb_image_logging.py` | 310 | ocr.domains.detection.utils.polygons<br>ocr.domains.detection.callbacks.wandb<br>ocr.domains.detection.utils.polygons | Polygon (14) |
| `models/architectures/__init__.py` | 7 | ocr.domains.detection.models.architectures<br>ocr.domains.recognition.models | DBNet (2)<br>CRAFT (2) |
| `lightning/__init__.py` | 55 | ocr.domains.detection.module<br>ocr.domains.recognition.module | PARSeq (1) |
| `metrics/cleval_metric.py` | 250 | ocr.domains.detection.metrics.box_types<br>ocr.domains.detection.metrics.functional | Polygon (1) |
| `models/architecture.py` | 317 | Clean | PARSeq (3)<br>Polygon (1) |
| `models/__init__.py` | 24 | ocr.domains.recognition.models | PARSeq (3) |
| `validation.py` | 1156 | Clean | Polygon (46) |
| `analysis/validation/analyze_worst_images.py` | 95 | Clean | BBox (4) |
| `evaluation/__init__.py` | 6 | ocr.domains.detection.evaluation | None |
| `inference/crop_extractor.py` | 360 | Clean | Polygon (38) |
| `inference/postprocessing_pipeline.py` | 132 | Clean | Polygon (1) |
| `inference/preprocessing_pipeline.py` | 393 | Clean | Polygon (1) |
| `inference/preview_generator.py` | 241 | Clean | Polygon (2) |
| `inference/utils.py` | 42 | Clean | Polygon (1) |
| `infrastructure/agents/validation_agent.py` | 353 | Clean | Accuracy (2) |
| `interfaces/schemas.py` | 107 | Clean | Polygon (2) |
| `lightning/loggers/wandb_loggers.py` | 278 | Clean | Polygon (6) |
| `lightning/utils/prediction_utils.py` | 52 | Clean | Polygon (2) |
| `models/architectures/shared_decoders.py` | 15 | ocr.domains.detection.models.decoders.fpn_decoder | None |
| `models/decoder/pan_decoder.py` | 99 | Clean | DBNet (1) |
| `models/decoder/unet.py` | 95 | Clean | DBNet (2) |
| `models/encoder/timm_backbone.py` | 151 | Clean | CRAFT (1) |
| `models/loss/cross_entropy_loss.py` | 63 | Clean | PARSeq (2) |
| `utils/background_normalization.py` | 46 | Clean | Accuracy (1) |
| `utils/config_utils.py` | 281 | Clean | DBNet (1) |
| `utils/convert_submission.py` | 143 | Clean | Polygon (2) |
| `utils/orientation.py` | 238 | Clean | Polygon (10) |
| `utils/sepia_enhancement.py` | 111 | Clean | Accuracy (1) |
