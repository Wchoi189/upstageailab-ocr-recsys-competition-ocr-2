ocr/
├── agents
│   ├── base_agent.py
│   ├── coordinator_agent.py
│   ├── __init__.py
│   ├── linting_agent.py
│   ├── llm
│   │   ├── base_client.py
│   │   ├── grok_client.py
│   │   ├── __init__.py
│   │   ├── openai_client.py
│   │   └── qwen_client.py
│   ├── ocr_agent.py
│   ├── __pycache__
│   │   ├── base_agent.cpython-311.pyc
│   │   ├── __init__.cpython-311.pyc
│   │   └── linting_agent.cpython-311.pyc
│   └── validation_agent.py
├── command_builder
│   ├── compute.py
│   ├── models.py
│   ├── overrides.py
│   ├── __pycache__
│   │   ├── models.cpython-311.pyc
│   │   └── recommendations.cpython-311.pyc
│   └── recommendations.py
├── communication
│   ├── __pycache__
│   │   ├── rabbitmq_transport.cpython-311.pyc
│   │   └── slack_service.cpython-311.pyc
│   ├── rabbitmq_transport.py
│   └── slack_service.py
├── core
│   ├── analysis
│   │   ├── data
│   │   │   ├── calculate_normalization.py
│   │   │   ├── gen_image_metadata.py
│   │   │   ├── __init__.py
│   │   │   └── recommend_buckets.py
│   │   ├── debugging
│   │   │   └── __init__.py
│   │   ├── __init__.py
│   │   └── validation
│   │       ├── analyze_worst_images.py
│   │       ├── __init__.py
│   │       └── render_underperforming.py
│   ├── evaluation
│   │   ├── evaluator.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       ├── evaluator.cpython-311.pyc
│   │       ├── evaluator.cpython-312.pyc
│   │       ├── __init__.cpython-311.pyc
│   │       └── __init__.cpython-312.pyc
│   ├── experiment.py
│   ├── inference
│   │   ├── config_loader.py
│   │   ├── coordinate_manager.py
│   │   ├── crop_extractor.py
│   │   ├── dependencies.py
│   │   ├── engine.py
│   │   ├── image_loader.py
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   ├── model_manager.py
│   │   ├── orchestrator.py
│   │   ├── postprocessing_pipeline.py
│   │   ├── postprocess.py
│   │   ├── preprocessing_metadata.py
│   │   ├── preprocessing_pipeline.py
│   │   ├── preprocess.py
│   │   ├── preview_generator.py
│   │   ├── __pycache__
│   │   │   ├── config_loader.cpython-311.pyc
│   │   │   ├── config_loader.cpython-312.pyc
│   │   │   ├── coordinate_manager.cpython-311.pyc
│   │   │   ├── crop_extractor.cpython-311.pyc
│   │   │   ├── dependencies.cpython-311.pyc
│   │   │   ├── dependencies.cpython-312.pyc
│   │   │   ├── engine.cpython-311.pyc
│   │   │   ├── engine.cpython-312.pyc
│   │   │   ├── image_loader.cpython-311.pyc
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── __init__.cpython-312.pyc
│   │   │   ├── model_loader.cpython-311.pyc
│   │   │   ├── model_loader.cpython-312.pyc
│   │   │   ├── model_manager.cpython-311.pyc
│   │   │   ├── orchestrator.cpython-311.pyc
│   │   │   ├── postprocess.cpython-311.pyc
│   │   │   ├── postprocess.cpython-312.pyc
│   │   │   ├── postprocessing_pipeline.cpython-311.pyc
│   │   │   ├── preprocess.cpython-311.pyc
│   │   │   ├── preprocess.cpython-312.pyc
│   │   │   ├── preprocessing_metadata.cpython-311.pyc
│   │   │   ├── preprocessing_pipeline.cpython-311.pyc
│   │   │   ├── preview_generator.cpython-311.pyc
│   │   │   ├── recognizer.cpython-311.pyc
│   │   │   ├── utils.cpython-311.pyc
│   │   │   └── utils.cpython-312.pyc
│   │   └── utils.py
│   ├── __init__.py
│   ├── interfaces
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── models.py
│   │   └── __pycache__
│   │       ├── losses.cpython-311.pyc
│   │       ├── metrics.cpython-311.pyc
│   │       └── models.cpython-311.pyc
│   ├── lightning
│   │   ├── callbacks
│   │   │   ├── __init__.py
│   │   │   ├── metadata_callback.py
│   │   │   ├── multi_line_progress_bar.py
│   │   │   ├── performance_profiler.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   ├── __init__.cpython-312.pyc
│   │   │   │   ├── kie_wandb_image_logging.cpython-311.pyc
│   │   │   │   ├── metadata_callback.cpython-311.pyc
│   │   │   │   ├── metadata_callback.cpython-312.pyc
│   │   │   │   ├── multi_line_progress_bar.cpython-311.pyc
│   │   │   │   ├── multi_line_progress_bar.cpython-312.pyc
│   │   │   │   ├── performance_profiler.cpython-311.pyc
│   │   │   │   ├── performance_profiler.cpython-312.pyc
│   │   │   │   ├── unique_checkpoint.cpython-311.pyc
│   │   │   │   ├── unique_checkpoint.cpython-312.pyc
│   │   │   │   ├── wandb_completion.cpython-311.pyc
│   │   │   │   ├── wandb_completion.cpython-312.pyc
│   │   │   │   ├── wandb_image_logging.cpython-311.pyc
│   │   │   │   └── wandb_image_logging.cpython-312.pyc
│   │   │   ├── unique_checkpoint.py
│   │   │   ├── wandb_completion.py
│   │   │   └── wandb_image_logging.py
│   │   ├── __init__.py
│   │   ├── loggers
│   │   │   ├── __init__.py
│   │   │   ├── progress_logger.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   ├── __init__.cpython-312.pyc
│   │   │   │   ├── wandb_loggers.cpython-311.pyc
│   │   │   │   └── wandb_loggers.cpython-312.pyc
│   │   │   └── wandb_loggers.py
│   │   ├── ocr_pl.py
│   │   ├── processors
│   │   │   ├── image_processor.py
│   │   │   ├── __init__.py
│   │   │   └── __pycache__
│   │   │       ├── image_processor.cpython-311.pyc
│   │   │       ├── image_processor.cpython-312.pyc
│   │   │       ├── __init__.cpython-311.pyc
│   │   │       └── __init__.cpython-312.pyc
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   └── ocr_pl.cpython-311.pyc
│   │   └── utils
│   │       ├── checkpoint_utils.py
│   │       ├── config_utils.py
│   │       ├── __init__.py
│   │       ├── model_utils.py
│   │       ├── prediction_utils.py
│   │       └── __pycache__
│   │           ├── checkpoint_utils.cpython-311.pyc
│   │           ├── checkpoint_utils.cpython-312.pyc
│   │           ├── config_utils.cpython-311.pyc
│   │           ├── config_utils.cpython-312.pyc
│   │           ├── __init__.cpython-311.pyc
│   │           ├── __init__.cpython-312.pyc
│   │           ├── prediction_utils.cpython-311.pyc
│   │           └── prediction_utils.cpython-312.pyc
│   ├── losses
│   │   └── __pycache__
│   │       └── base.cpython-311.pyc
│   ├── metrics
│   │   ├── box_types.py
│   │   ├── cleval_metric.py
│   │   ├── data.py
│   │   ├── eval_functions.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── base.cpython-311.pyc
│   │   │   ├── box_types.cpython-311.pyc
│   │   │   ├── cleval_metric.cpython-311.pyc
│   │   │   ├── data.cpython-311.pyc
│   │   │   ├── eval_functions.cpython-311.pyc
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   └── utils.cpython-311.pyc
│   │   ├── README.md
│   │   └── utils.py
│   ├── models
│   │   ├── architecture.py
│   │   ├── architectures
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── craft.cpython-311.pyc
│   │   │   │   ├── craft.cpython-312.pyc
│   │   │   │   ├── dbnet.cpython-311.pyc
│   │   │   │   ├── dbnet.cpython-312.pyc
│   │   │   │   ├── dbnetpp.cpython-311.pyc
│   │   │   │   ├── dbnetpp.cpython-312.pyc
│   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   ├── __init__.cpython-312.pyc
│   │   │   │   ├── parseq.cpython-311.pyc
│   │   │   │   ├── shared_decoders.cpython-311.pyc
│   │   │   │   └── shared_decoders.cpython-312.pyc
│   │   │   └── shared_decoders.py
│   │   ├── core
│   │   │   └── __pycache__
│   │   │       ├── base_classes.cpython-311.pyc
│   │   │       ├── base_classes.cpython-312.pyc
│   │   │       ├── __init__.cpython-311.pyc
│   │   │       ├── __init__.cpython-312.pyc
│   │   │       ├── registry.cpython-311.pyc
│   │   │       └── registry.cpython-312.pyc
│   │   ├── decoder
│   │   │   ├── __init__.py
│   │   │   ├── pan_decoder.py
│   │   │   ├── __pycache__
│   │   │   │   ├── craft_decoder.cpython-311.pyc
│   │   │   │   ├── craft_decoder.cpython-312.pyc
│   │   │   │   ├── dbpp_decoder.cpython-311.pyc
│   │   │   │   ├── dbpp_decoder.cpython-312.pyc
│   │   │   │   ├── fpn_decoder.cpython-311.pyc
│   │   │   │   ├── fpn_decoder.cpython-312.pyc
│   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   ├── __init__.cpython-312.pyc
│   │   │   │   ├── pan_decoder.cpython-311.pyc
│   │   │   │   ├── pan_decoder.cpython-312.pyc
│   │   │   │   ├── parseq_decoder.cpython-311.pyc
│   │   │   │   ├── unet.cpython-311.pyc
│   │   │   │   └── unet.cpython-312.pyc
│   │   │   └── unet.py
│   │   ├── encoder
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── craft_vgg.cpython-311.pyc
│   │   │   │   ├── craft_vgg.cpython-312.pyc
│   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   ├── __init__.cpython-312.pyc
│   │   │   │   ├── timm_backbone.cpython-311.pyc
│   │   │   │   └── timm_backbone.cpython-312.pyc
│   │   │   └── timm_backbone.py
│   │   ├── head
│   │   │   ├── __init__.py
│   │   │   └── __pycache__
│   │   │       ├── craft_head.cpython-311.pyc
│   │   │       ├── craft_head.cpython-312.pyc
│   │   │       ├── craft_postprocess.cpython-311.pyc
│   │   │       ├── craft_postprocess.cpython-312.pyc
│   │   │       ├── db_head.cpython-311.pyc
│   │   │       ├── db_head.cpython-312.pyc
│   │   │       ├── db_postprocess.cpython-311.pyc
│   │   │       ├── db_postprocess.cpython-312.pyc
│   │   │       ├── __init__.cpython-311.pyc
│   │   │       ├── __init__.cpython-312.pyc
│   │   │       └── parseq_head.cpython-311.pyc
│   │   ├── __init__.py
│   │   ├── layers
│   │   │   ├── common.py
│   │   │   └── __pycache__
│   │   │       └── common.cpython-311.pyc
│   │   ├── loss
│   │   │   ├── bce_loss.py
│   │   │   ├── craft_loss.py
│   │   │   ├── cross_entropy_loss.py
│   │   │   ├── db_loss.py
│   │   │   ├── dice_loss.py
│   │   │   ├── __init__.py
│   │   │   ├── l1_loss.py
│   │   │   └── __pycache__
│   │   │       ├── bce_loss.cpython-311.pyc
│   │   │       ├── bce_loss.cpython-312.pyc
│   │   │       ├── craft_loss.cpython-311.pyc
│   │   │       ├── craft_loss.cpython-312.pyc
│   │   │       ├── cross_entropy_loss.cpython-311.pyc
│   │   │       ├── db_loss.cpython-311.pyc
│   │   │       ├── db_loss.cpython-312.pyc
│   │   │       ├── dice_loss.cpython-311.pyc
│   │   │       ├── dice_loss.cpython-312.pyc
│   │   │       ├── __init__.cpython-311.pyc
│   │   │       ├── __init__.cpython-312.pyc
│   │   │       ├── l1_loss.cpython-311.pyc
│   │   │       └── l1_loss.cpython-312.pyc
│   │   └── __pycache__
│   │       ├── architecture.cpython-311.pyc
│   │       ├── architecture.cpython-312.pyc
│   │       ├── __init__.cpython-311.pyc
│   │       ├── __init__.cpython-312.pyc
│   │       ├── interfaces.cpython-311.pyc
│   │       └── kie_models.cpython-311.pyc
│   ├── __pycache__
│   │   ├── architecture.cpython-311.pyc
│   │   ├── base_classes.cpython-311.pyc
│   │   ├── __init__.cpython-311.pyc
│   │   ├── kie_validation.cpython-311.pyc
│   │   ├── registry.cpython-311.pyc
│   │   └── validation.cpython-311.pyc
│   ├── utils
│   │   ├── api_usage_tracker.py
│   │   ├── background_normalization.py
│   │   ├── cache_manager.py
│   │   ├── callbacks.py
│   │   ├── checkpoints
│   │   │   ├── __init__.py
│   │   │   ├── metadata_loader.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   ├── metadata_loader.cpython-311.pyc
│   │   │   │   └── types.cpython-311.pyc
│   │   │   └── types.py
│   │   ├── command
│   │   │   ├── builder.py
│   │   │   ├── executor.py
│   │   │   ├── __init__.py
│   │   │   ├── models.py
│   │   │   ├── __pycache__
│   │   │   │   ├── builder.cpython-311.pyc
│   │   │   │   ├── builder.cpython-312.pyc
│   │   │   │   ├── executor.cpython-311.pyc
│   │   │   │   ├── executor.cpython-312.pyc
│   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   ├── __init__.cpython-312.pyc
│   │   │   │   ├── models.cpython-311.pyc
│   │   │   │   ├── models.cpython-312.pyc
│   │   │   │   ├── quoting.cpython-311.pyc
│   │   │   │   ├── quoting.cpython-312.pyc
│   │   │   │   ├── validator.cpython-311.pyc
│   │   │   │   └── validator.cpython-312.pyc
│   │   │   ├── quoting.py
│   │   │   └── validator.py
│   │   ├── config.py
│   │   ├── config_utils.py
│   │   ├── config_validation.py
│   │   ├── convert_submission.py
│   │   ├── data_utils.py
│   │   ├── experiment_index.py
│   │   ├── experiment_name.py
│   │   ├── geometry_utils.py
│   │   ├── image_loading.py
│   │   ├── image_utils.py
│   │   ├── __init__.py
│   │   ├── logger_factory.py
│   │   ├── logging.py
│   │   ├── ocr_utils.py
│   │   ├── orientation_constants.py
│   │   ├── orientation.py
│   │   ├── path_utils.py
│   │   ├── perspective_correction
│   │   │   ├── core.py
│   │   │   ├── fitting.py
│   │   │   ├── geometry.py
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── core.cpython-311.pyc
│   │   │   │   ├── fitting.cpython-311.pyc
│   │   │   │   ├── geometry.cpython-311.pyc
│   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   ├── quality_metrics.cpython-311.pyc
│   │   │   │   ├── types.cpython-311.pyc
│   │   │   │   └── validation.cpython-311.pyc
│   │   │   ├── quality_metrics.py
│   │   │   ├── types.py
│   │   │   └── validation.py
│   │   ├── polygon_utils.py
│   │   ├── __pycache__
│   │   │   ├── background_normalization.cpython-311.pyc
│   │   │   ├── cache_manager.cpython-311.pyc
│   │   │   ├── callbacks.cpython-311.pyc
│   │   │   ├── config_utils.cpython-311.pyc
│   │   │   ├── data_utils.cpython-311.pyc
│   │   │   ├── experiment_index.cpython-311.pyc
│   │   │   ├── geometry_utils.cpython-311.pyc
│   │   │   ├── image_loading.cpython-311.pyc
│   │   │   ├── image_utils.cpython-311.pyc
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── logger_factory.cpython-311.pyc
│   │   │   ├── logging.cpython-311.pyc
│   │   │   ├── orientation_constants.cpython-311.pyc
│   │   │   ├── orientation.cpython-311.pyc
│   │   │   ├── path_utils.cpython-311.pyc
│   │   │   ├── polygon_utils.cpython-311.pyc
│   │   │   ├── registry.cpython-311.pyc
│   │   │   ├── sepia_enhancement.cpython-311.pyc
│   │   │   ├── submission.cpython-311.pyc
│   │   │   ├── text_rendering.cpython-311.pyc
│   │   │   └── wandb_utils.cpython-311.pyc
│   │   ├── registry.py
│   │   ├── sepia_enhancement.py
│   │   ├── submission.py
│   │   ├── text_rendering.py
│   │   └── wandb_utils.py
│   └── validation.py
├── data
│   ├── charset.json
│   ├── datasets
│   │   ├── base.py
│   │   ├── craft_collate_fn.py
│   │   ├── db_collate_fn.py
│   │   ├── __init__.py
│   │   ├── preprocessing
│   │   │   ├── advanced_detector.py
│   │   │   ├── advanced_noise_elimination.py
│   │   │   ├── advanced_preprocessor.py
│   │   │   ├── archive
│   │   │   │   └── phase1_experimental_modules
│   │   │   │       └── README.md
│   │   │   ├── background_removal.py
│   │   │   ├── config.py
│   │   │   ├── contracts.py
│   │   │   ├── detector.py
│   │   │   ├── document_flattening.py
│   │   │   ├── enhanced_pipeline.py
│   │   │   ├── enhancement.py
│   │   │   ├── external.py
│   │   │   ├── __init__.py
│   │   │   ├── intelligent_brightness.py
│   │   │   ├── metadata.py
│   │   │   ├── orientation.py
│   │   │   ├── padding.py
│   │   │   ├── perspective.py
│   │   │   ├── pipeline.py
│   │   │   ├── __pycache__
│   │   │   │   ├── advanced_corner_detection.cpython-311.pyc
│   │   │   │   ├── advanced_corner_detection.cpython-312.pyc
│   │   │   │   ├── advanced_detector.cpython-311.pyc
│   │   │   │   ├── advanced_detector.cpython-312.pyc
│   │   │   │   ├── advanced_detector_test.cpython-311-pytest-9.0.1.pyc
│   │   │   │   ├── advanced_noise_elimination.cpython-311.pyc
│   │   │   │   ├── advanced_noise_elimination.cpython-312.pyc
│   │   │   │   ├── advanced_preprocessor.cpython-311.pyc
│   │   │   │   ├── advanced_preprocessor.cpython-312.pyc
│   │   │   │   ├── background_removal.cpython-311.pyc
│   │   │   │   ├── background_removal.cpython-312.pyc
│   │   │   │   ├── config.cpython-311.pyc
│   │   │   │   ├── config.cpython-312.pyc
│   │   │   │   ├── contracts.cpython-311.pyc
│   │   │   │   ├── contracts.cpython-312.pyc
│   │   │   │   ├── detector.cpython-311.pyc
│   │   │   │   ├── detector.cpython-312.pyc
│   │   │   │   ├── document_flattening.cpython-311.pyc
│   │   │   │   ├── document_flattening.cpython-312.pyc
│   │   │   │   ├── enhanced_pipeline.cpython-311.pyc
│   │   │   │   ├── enhanced_pipeline.cpython-312.pyc
│   │   │   │   ├── enhancement.cpython-311.pyc
│   │   │   │   ├── enhancement.cpython-312.pyc
│   │   │   │   ├── external.cpython-311.pyc
│   │   │   │   ├── external.cpython-312.pyc
│   │   │   │   ├── geometric_document_modeling.cpython-311.pyc
│   │   │   │   ├── geometric_document_modeling.cpython-312.pyc
│   │   │   │   ├── high_confidence_decision_making.cpython-311.pyc
│   │   │   │   ├── high_confidence_decision_making.cpython-312.pyc
│   │   │   │   ├── __init__.cpython-311.pyc
│   │   │   │   ├── __init__.cpython-312.pyc
│   │   │   │   ├── intelligent_brightness.cpython-311.pyc
│   │   │   │   ├── intelligent_brightness.cpython-312.pyc
│   │   │   │   ├── metadata.cpython-311.pyc
│   │   │   │   ├── metadata.cpython-312.pyc
│   │   │   │   ├── orientation.cpython-311.pyc
│   │   │   │   ├── orientation.cpython-312.pyc
│   │   │   │   ├── padding.cpython-311.pyc
│   │   │   │   ├── padding.cpython-312.pyc
│   │   │   │   ├── perspective.cpython-311.pyc
│   │   │   │   ├── perspective.cpython-312.pyc
│   │   │   │   ├── pipeline.cpython-311.pyc
│   │   │   │   ├── pipeline.cpython-312.pyc
│   │   │   │   ├── resize.cpython-311.pyc
│   │   │   │   ├── resize.cpython-312.pyc
│   │   │   │   ├── validators.cpython-311.pyc
│   │   │   │   └── validators.cpython-312.pyc
│   │   │   ├── resize.py
│   │   │   ├── telemetry.py
│   │   │   └── validators.py
│   │   ├── __pycache__
│   │   │   ├── base.cpython-311.pyc
│   │   │   ├── db_collate_fn.cpython-311.pyc
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── kie_dataset.cpython-311.pyc
│   │   │   ├── recognition_collate_fn.cpython-311.pyc
│   │   │   ├── schemas.cpython-311.pyc
│   │   │   └── transforms.cpython-311.pyc
│   │   ├── recognition_collate_fn.py
│   │   ├── schemas.py
│   │   └── transforms.py
│   ├── __pycache__
│   │   └── tokenizer.cpython-311.pyc
│   └── schemas
│       ├── __pycache__
│       │   └── storage.cpython-311.pyc
│       └── storage.py
├── experiment_registry.py
├── features
│   ├── detection
│   │   ├── __init__.py
│   │   ├── interfaces.py
│   │   ├── models
│   │   │   ├── architectures
│   │   │   │   ├── craft.py
│   │   │   │   ├── dbnetpp.py
│   │   │   │   ├── dbnet.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── __pycache__
│   │   │   │       ├── craft.cpython-311.pyc
│   │   │   │       ├── dbnet.cpython-311.pyc
│   │   │   │       ├── dbnetpp.cpython-311.pyc
│   │   │   │       └── __init__.cpython-311.pyc
│   │   │   ├── decoders
│   │   │   │   ├── craft_decoder.py
│   │   │   │   ├── dbpp_decoder.py
│   │   │   │   ├── fpn_decoder.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── __pycache__
│   │   │   │       ├── craft_decoder.cpython-311.pyc
│   │   │   │       ├── dbpp_decoder.cpython-311.pyc
│   │   │   │       ├── fpn_decoder.cpython-311.pyc
│   │   │   │       └── __init__.cpython-311.pyc
│   │   │   ├── encoders
│   │   │   │   ├── craft_vgg.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── __pycache__
│   │   │   │       ├── craft_vgg.cpython-311.pyc
│   │   │   │       └── __init__.cpython-311.pyc
│   │   │   ├── heads
│   │   │   │   ├── craft_head.py
│   │   │   │   ├── db_head.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── __pycache__
│   │   │   │       ├── craft_head.cpython-311.pyc
│   │   │   │       ├── db_head.cpython-311.pyc
│   │   │   │       └── __init__.cpython-311.pyc
│   │   │   ├── __init__.py
│   │   │   ├── postprocess
│   │   │   │   ├── craft_postprocess.py
│   │   │   │   ├── db_postprocess.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── __pycache__
│   │   │   │       ├── craft_postprocess.cpython-311.pyc
│   │   │   │       ├── db_postprocess.cpython-311.pyc
│   │   │   │       └── __init__.cpython-311.pyc
│   │   │   └── __pycache__
│   │   │       └── __init__.cpython-311.pyc
│   │   └── __pycache__
│   │       ├── __init__.cpython-311.pyc
│   │       └── interfaces.cpython-311.pyc
│   ├── kie
│   │   ├── data
│   │   │   ├── dataset.py
│   │   │   ├── __init__.py
│   │   │   └── __pycache__
│   │   │       ├── dataset.cpython-311.pyc
│   │   │       └── __init__.cpython-311.pyc
│   │   ├── inference
│   │   │   └── extraction
│   │   │       ├── field_extractor.py
│   │   │       ├── __init__.py
│   │   │       ├── normalizers.py
│   │   │       ├── __pycache__
│   │   │       │   ├── field_extractor.cpython-311.pyc
│   │   │       │   ├── __init__.cpython-311.pyc
│   │   │       │   ├── normalizers.cpython-311.pyc
│   │   │       │   └── receipt_schema.cpython-311.pyc
│   │   │       ├── receipt_schema.py
│   │   │       └── vlm_extractor.py
│   │   ├── __init__.py
│   │   ├── lightning
│   │   │   ├── callbacks
│   │   │   │   ├── __init__.py
│   │   │   │   └── kie_wandb_image_logging.py
│   │   │   └── __init__.py
│   │   ├── models
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   └── __pycache__
│   │   │       ├── __init__.cpython-311.pyc
│   │   │       └── model.cpython-311.pyc
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── trainer.cpython-311.pyc
│   │   │   └── validation.cpython-311.pyc
│   │   ├── trainer.py
│   │   └── validation.py
│   ├── layout
│   │   ├── inference
│   │   │   ├── contracts.py
│   │   │   ├── grouper.py
│   │   │   ├── __init__.py
│   │   │   └── __pycache__
│   │   │       ├── contracts.cpython-311.pyc
│   │   │       ├── grouper.cpython-311.pyc
│   │   │       └── __init__.cpython-311.pyc
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   └── __init__.cpython-311.pyc
│   │   └── README.md
│   └── recognition
│       ├── callbacks
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── __init__.cpython-311.pyc
│       │   │   └── wandb_image_logging.cpython-311.pyc
│       │   └── wandb_image_logging.py
│       ├── data
│       │   ├── __init__.py
│       │   ├── lmdb_dataset.py
│       │   ├── __pycache__
│       │   │   ├── __init__.cpython-311.pyc
│       │   │   ├── lmdb_dataset.cpython-311.pyc
│       │   │   └── tokenizer.cpython-311.pyc
│       │   └── tokenizer.py
│       ├── inference
│       │   ├── backends
│       │   │   ├── __init__.py
│       │   │   ├── paddleocr_recognizer.py
│       │   │   └── __pycache__
│       │   │       ├── __init__.cpython-311.pyc
│       │   │       └── paddleocr_recognizer.cpython-311.pyc
│       │   ├── __pycache__
│       │   │   └── recognizer.cpython-311.pyc
│       │   └── recognizer.py
│       └── models
│           ├── architecture.py
│           ├── decoder.py
│           ├── head.py
│           ├── __init__.py
│           └── __pycache__
│               ├── architecture.cpython-311.pyc
│               ├── decoder.cpython-311.pyc
│               ├── head.cpython-311.pyc
│               └── __init__.cpython-311.pyc
├── __init__.py
├── __pycache__
│   ├── __init__.cpython-310.pyc
│   ├── __init__.cpython-311.pyc
│   └── __init__.cpython-312.pyc
├── synthetic_data
│   ├── dataset.py
│   ├── generators
│   │   ├── background.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── background.cpython-311.pyc
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   ├── renderer.cpython-311.pyc
│   │   │   └── text.cpython-311.pyc
│   │   ├── renderer.py
│   │   └── text.py
│   ├── __init__.py
│   ├── models.py
│   ├── __pycache__
│   │   ├── dataset.cpython-311.pyc
│   │   ├── __init__.cpython-311.pyc
│   │   ├── models.cpython-311.pyc
│   │   └── utils.cpython-311.pyc
│   └── utils.py
└── validation
    ├── models.py
    └── __pycache__
        └── models.cpython-311.pyc

115 directories, 542 files
