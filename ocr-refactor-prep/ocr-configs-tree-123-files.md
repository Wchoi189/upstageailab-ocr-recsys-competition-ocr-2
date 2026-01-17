configs/
├── base.yaml
├── data
│   ├── base.yaml
│   ├── canonical.yaml
│   ├── craft.yaml
│   ├── dataloaders
│   │   ├── default.yaml
│   │   └── rtx3060_16core.yaml
│   ├── datasets
│   │   ├── db.yaml
│   │   ├── preprocessing_camscanner.yaml
│   │   ├── preprocessing_docTR_demo.yaml
│   │   └── preprocessing.yaml
│   ├── default.yaml
│   ├── performance_preset
│   │   ├── balanced.yaml
│   │   ├── memory_efficient.yaml
│   │   ├── minimal.yaml
│   │   ├── none.yaml
│   │   ├── README.md
│   │   └── validation_optimized.yaml
│   ├── recognition.yaml
│   └── transforms
│       ├── background_removal.yaml
│       ├── base.yaml
│       ├── recognition.yaml
│       └── with_background_removal.yaml
├── debug
│   └── default.yaml
├── domain
│   ├── detection.yaml
│   ├── kie.yaml
│   ├── layout.yaml
│   └── recognition.yaml
├── evaluation
│   └── metrics.yaml
├── eval.yaml
├── __EXTENDED__
│   ├── benchmarks
│   │   └── decoder.yaml
│   ├── examples
│   │   ├── predict_full_enhancement.yaml
│   │   ├── predict_shadow_removal.yaml
│   │   ├── predict_with_perspective.yaml
│   │   └── README.md
│   ├── experiments
│   │   └── train_v2.yaml
│   └── kie_variants
│       ├── train_kie_aihub_only.yaml
│       ├── train_kie_aihub_production.yaml
│       ├── train_kie_aihub.yaml
│       ├── train_kie_baseline_optimized_v2.yaml
│       ├── train_kie_merged_3090_10ep.yaml
│       ├── train_kie.yaml
│       └── train_parseq.yaml
├── extraction
│   ├── default.yaml
│   └── hybrid.yaml
├── _foundation
│   ├── core.yaml
│   ├── data.yaml
│   ├── defaults.yaml
│   ├── logging.yaml
│   ├── model.yaml
│   ├── preprocessing.yaml
│   └── trainer.yaml
├── hydra
│   ├── default.yaml
│   └── disabled.yaml
├── layout
│   └── default.yaml
├── __LEGACY__
│   ├── data
│   │   └── preprocessing.yaml
│   ├── model
│   │   └── optimizer.yaml
│   ├── README_20260108_deprecated.md
│   └── README.md
├── model
│   ├── architectures
│   │   ├── craft.yaml
│   │   ├── dbnetpp.yaml
│   │   ├── dbnet.yaml
│   │   └── parseq.yaml
│   ├── decoder
│   │   ├── craft_decoder.yaml
│   │   ├── dbpp_decoder.yaml
│   │   ├── fpn.yaml
│   │   ├── pan.yaml
│   │   └── unet.yaml
│   ├── default.yaml
│   ├── encoder
│   │   ├── craft_vgg.yaml
│   │   └── timm_backbone.yaml
│   ├── head
│   │   ├── craft_head.yaml
│   │   ├── db_head.yaml
│   │   └── dbpp_head.yaml
│   ├── lightning_modules
│   │   └── base.yaml
│   ├── loss
│   │   ├── craft_loss.yaml
│   │   └── db_loss.yaml
│   ├── optimizers
│   │   ├── adamw.yaml
│   │   └── adam.yaml
│   ├── presets
│   │   ├── craft.yaml
│   │   ├── dbnetpp.yaml
│   │   └── model_example.yaml
│   └── recognition.yaml
├── paths
│   └── default.yaml
├── predict.yaml
├── README.md
├── recognition
│   ├── default.yaml
│   └── paddleocr.yaml
├── synthetic.yaml
├── trainer
│   ├── debug_crash.yaml
│   ├── debug_safe.yaml
│   ├── default.yaml
│   ├── fp16_safe.yaml
│   ├── hardware_rtx3060_12gb_i5_16core.yaml
│   ├── hardware_rtx3090_24gb_i5_16core.yaml
│   ├── rtx3060_12gb.yaml
│   └── rtx3090_24gb.yaml
├── training
│   ├── callbacks
│   │   ├── default.yaml
│   │   ├── early_stopping.yaml
│   │   ├── metadata.yaml
│   │   ├── model_checkpoint.yaml
│   │   ├── model_summary.yaml
│   │   ├── performance_profiler.yaml
│   │   ├── recognition_wandb.yaml
│   │   ├── rich_progress_bar.yaml
│   │   └── wandb_image_logging.yaml
│   ├── logger
│   │   ├── architectures
│   │   │   ├── craft.yaml
│   │   │   ├── dbnetpp.yaml
│   │   │   └── dbnet.yaml
│   │   ├── consolidated.yaml
│   │   ├── csv.yaml
│   │   ├── default.yaml
│   │   ├── inference.yaml
│   │   ├── modes
│   │   │   ├── comparison.yaml
│   │   │   ├── inference.yaml
│   │   │   └── preprocessing.yaml
│   │   ├── optimizers
│   │   │   ├── adamw.yaml
│   │   │   └── adam.yaml
│   │   ├── preprocessing_profiles.yaml
│   │   ├── unified_app.yaml
│   │   └── wandb.yaml
│   └── profiling
│       ├── cache_performance_test.yaml
│       └── performance_test.yaml
└── train.yaml

39 directories, 123 files
