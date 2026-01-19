# Detection Model Component Paths - Quick Reference

## Architecture Structure
```
ocr/domains/detection/models/
├── architectures/      # Full model compositions (DBNet, CRAFT, etc.)
├── encoders/          # Backbone networks
├── decoders/          # Feature decoders
├── heads/             # Prediction heads
└── postprocess/       # Post-processing logic
```

## Correct `_target_` Paths

### Encoders
```yaml
encoder:
  _target_: ocr.core.models.encoder.TimmBackbone
  # or
  _target_: ocr.domains.detection.models.encoders.craft_vgg.CRAFTVGGEncoder
```

### Decoders
```yaml
decoder:
  _target_: ocr.domains.detection.models.decoders.fpn_decoder.FPNDecoder
  # or
  _target_: ocr.domains.detection.models.decoders.dbpp_decoder.DBPPDecoder
  # or
  _target_: ocr.domains.detection.models.decoders.craft_decoder.CRAFTDecoder
```

### Heads
```yaml
head:
  _target_: ocr.domains.detection.models.heads.db_head.DBHead
  # or
  _target_: ocr.domains.detection.models.heads.craft_head.CRAFTHead
```

## DBNet Configuration Example
```yaml
# @package model
_target_: ocr.core.models.architecture.OCRModel

encoder:
  _target_: ocr.core.models.encoder.TimmBackbone
  model_name: "resnet18"
  pretrained: true
  output_indices: [1, 2, 3, 4]

decoder:
  _target_: ocr.domains.detection.models.decoders.fpn_decoder.FPNDecoder
  in_channels: [64, 128, 256, 512]
  inner_channels: 256
  out_channels: 256

head:
  _target_: ocr.domains.detection.models.heads.db_head.DBHead
  in_channels: 256
  upscale: 4
  binarization_threshold: 0.3
  expand_ratio: 1.5
```

## Finding Correct Paths

```bash
# List all available components:
find ocr/domains/detection/models -name "*.py" -type f

# Find class definitions:
grep -r "^class " ocr/domains/detection/models/
```
