# Context Tree: /workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr

📁 **/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/**
  📁 **agents/**
    📁 **llm/** - *LLM client wrappers for multi-agent collaboration.*
      - Exports: BaseLLMClient, QwenClient, Grok4Client, OpenAIClient
      📄 `__init__.py`
      📄 `base_client.py` - Classes: LLMResponse, BaseLLMClient
      📄 `grok_client.py` - Classes: Grok4Client
      📄 `openai_client.py` - Classes: OpenAIClient
      📄 `qwen_client.py` - Classes: QwenClient
    📄 `__init__.py`
    📄 `base_agent.py` - Classes: AgentCapability, AgentMetadata, BaseAgent
    📄 `coordinator_agent.py` - Classes: CoordinatorAgent
    📄 `linting_agent.py` - Classes: LintingAgent
    📄 `ocr_agent.py` - Classes: OCRAgent
    📄 `validation_agent.py` - Classes: ValidationAgent
  📁 **command_builder/**
    📄 `compute.py` - Functions: compute_overrides
    📄 `models.py` - Classes: UseCaseRecommendation
    📄 `overrides.py` - Functions: build_additional_overrides, maybe_suffix_exp_name
    📄 `recommendations.py` - Classes: UseCaseRecommendationService
  📁 **communication/**
    📄 `rabbitmq_transport.py` - Classes: RabbitMQTransport
    📄 `slack_service.py` - Classes: SlackNotificationService
  📁 **core/** - *⚡ Core* - *Core abstract base classes and registry for OCR framework components.*
    - Exports: BaseEncoder, BaseDecoder, BaseHead, BaseLoss, BaseMetric, ... (8 total)
    📁 **analysis/** - *Analysis tools for OCR model debugging, validation, and data insights.*
      📁 **data/** - *📊 Data* - *Scripts for analyzing and preprocessing OCR training data.*
        📄 `__init__.py`
        📄 `calculate_normalization.py` - Functions: calculate_normalization_stats
        📄 `gen_image_metadata.py` - Functions: propose_bucket, main
        📄 `recommend_buckets.py` - Functions: main
      📁 **debugging/** - *Scripts for debugging and visualizing OCR model behavior.*
        📄 `__init__.py`
      📁 **validation/** - *Scripts for validating and evaluating OCR model predictions.*
        📄 `__init__.py`
        📄 `analyze_worst_images.py` - Functions: draw_bboxes_on_image, analyze_worst_images
        📄 `render_underperforming.py` - Functions: build_html_block, main
      📄 `__init__.py`
    📁 **evaluation/** - *Evaluation helpers for OCR Lightning modules.*
      - Exports: CLEvalEvaluator
      📄 `__init__.py`
      📄 `evaluator.py` - Classes: CLEvalEvaluator
    📁 **inference/** - *🔮 Inference* - *Modular helpers for OCR inference utilities.*
      - Exports: InferenceEngine, run_inference_on_image, get_available_checkpoints, ModelConfigBundle, PreprocessSettings, ... (6 total)
      📄 `__init__.py`
      📄 `config_loader.py` - Classes: NormalizationSettings, PreprocessSettings, PostprocessSettings - Functions: resolve_config_path, load_model_config
      📄 `coordinate_manager.py` - Classes: TransformMetadata, CoordinateTransformationManager - Functions: calculate_transform_metadata, compute_inverse_matrix, compute_forward_scales
      📄 `crop_extractor.py` - Classes: CropConfig, CropResult, CropExtractor
      📄 `dependencies.py`
      📄 `engine.py` - Classes: InferenceEngine - Functions: run_inference_on_image, get_available_checkpoints
      📄 `image_loader.py` - Classes: LoadedImage, ImageLoader
      📄 `model_loader.py` - Functions: instantiate_model, load_checkpoint, load_state_dict
      📄 `model_manager.py` - Classes: ModelManager
      📄 `orchestrator.py` - Classes: InferenceOrchestrator
      📄 `postprocess.py` - Functions: compute_inverse_matrix, decode_polygons_with_head, fallback_postprocess
      📄 `postprocessing_pipeline.py` - Classes: PostprocessingResult, PostprocessingPipeline
      📄 `preprocess.py` - Functions: build_transform, preprocess_image, apply_optional_perspective_correction
      📄 `preprocessing_metadata.py` - Functions: create_preprocessing_metadata, calculate_resize_dimensions, calculate_padding
      📄 `preprocessing_pipeline.py` - Classes: PreprocessingResult, PreprocessingPipeline
      📄 `preview_generator.py` - Classes: PreviewGenerator - Functions: create_preview_with_metadata
      📄 `utils.py` - Functions: get_available_checkpoints, generate_mock_predictions, ensure_three_channel
    📁 **interfaces/**
      📄 `losses.py` - Classes: BaseLoss
      📄 `metrics.py` - Classes: BaseMetric
      📄 `models.py` - Classes: BaseEncoder, BaseDecoder, BaseHead
    📁 **lightning/**
      📁 **callbacks/**
        - Exports: MetadataCallback, PerformanceProfilerCallback
        📄 `__init__.py`
        📄 `metadata_callback.py` - Classes: MetadataCallback
        📄 `multi_line_progress_bar.py` - Classes: MultiLineRichProgressBar
        📄 `performance_profiler.py` - Classes: PerformanceProfilerCallback
        📄 `unique_checkpoint.py` - Classes: UniqueModelCheckpoint
        📄 `wandb_completion.py` - Classes: WandbCompletionCallback
        📄 `wandb_image_logging.py` - Classes: WandbImageLoggingCallback
      📁 **loggers/**
        - Exports: get_rich_console, WandbProblemLogger
        📄 `__init__.py`
        📄 `progress_logger.py`
        📄 `wandb_loggers.py` - Classes: WandbProblemLogger
      📁 **processors/**
        - Exports: ImageProcessor
        📄 `__init__.py`
        📄 `image_processor.py` - Classes: ImageProcessor
      📁 **utils/** - *🔧 Utils*
        - Exports: extract_metric_kwargs, extract_normalize_stats, CheckpointHandler, format_predictions
        📄 `__init__.py`
        📄 `checkpoint_utils.py` - Classes: CheckpointHandler
        📄 `config_utils.py` - Functions: extract_metric_kwargs, extract_normalize_stats
        📄 `model_utils.py` - Functions: load_state_dict_with_fallback
        📄 `prediction_utils.py` - Functions: format_predictions
      📄 `__init__.py` - Functions: get_pl_modules_by_cfg
      📄 `ocr_pl.py` - Classes: OCRPLModule, OCRDataPLModule
    📁 **losses/**
    📁 **metrics/**
      📄 `README.md`
      📄 `__init__.py`
      📄 `box_types.py` - Classes: Box, QUAD, POLY - Functions: get_midpoints, point_distance, unit_vector
      📄 `cleval_metric.py` - Classes: Options, CLEvalMetric
      📄 `data.py` - Classes: MatchReleation, CoreStats, MatchResult - Functions: accumulate_result, accumulate_stats, accumulate_core_stats
      📄 `eval_functions.py` - Classes: EvalMaterial - Functions: evaluation, prepare_gt, prepare_det
      📄 `utils.py` - Functions: load_zip_file, decode_utf8, dump_json
    📁 **models/** - *🤖 Models*
      📁 **architectures/** - *OCR architecture implementations and registrations.*
        - Exports: dbnet, craft, dbnetpp, shared_decoders, recognition_arch
        📄 `__init__.py`
        📄 `shared_decoders.py` - Functions: register_shared_decoders
      📁 **core/** - *⚡ Core*
      📁 **decoder/**
        📄 `__init__.py` - Functions: get_decoder_by_cfg
        📄 `pan_decoder.py` - Classes: PANDecoder
        📄 `unet.py` - Classes: UNetDecoder
      📁 **encoder/**
        📄 `__init__.py` - Functions: get_encoder_by_cfg
        📄 `timm_backbone.py` - Classes: TimmBackbone
      📁 **head/**
        📄 `__init__.py` - Functions: get_head_by_cfg
      📁 **layers/**
        📄 `common.py` - Functions: conv_bn_relu
      📁 **loss/**
        📄 `__init__.py` - Functions: get_loss_by_cfg
        📄 `bce_loss.py` - Classes: BCELoss
        📄 `craft_loss.py` - Classes: CraftLoss
        📄 `cross_entropy_loss.py` - Classes: CrossEntropyLoss
        📄 `db_loss.py` - Classes: DBLoss
        📄 `dice_loss.py` - Classes: DiceLoss
        📄 `l1_loss.py` - Classes: MaskL1Loss
      📄 `__init__.py` - Functions: get_model_by_cfg
      📄 `architecture.py` - Classes: OCRModel
    📁 **utils/** - *🔧 Utils*
      📁 **checkpoints/**
        📄 `__init__.py`
        📄 `metadata_loader.py` - Functions: save_metadata
        📄 `types.py` - Classes: TrainingInfo, EncoderInfo, DecoderInfo
      📁 **command/** - *Command Utilities Package*
        - Exports: CommandBuilder, CommandExecutor, CommandValidator
        📄 `__init__.py`
        📄 `builder.py` - Classes: CommandBuilder
        📄 `executor.py` - Classes: CommandExecutor
        📄 `models.py` - Classes: CommandParams, TrainCommandParams, TestCommandParams
        📄 `quoting.py` - Functions: quote_override, is_special_char
        📄 `validator.py` - Classes: CommandValidator
      📁 **perspective_correction/** - *Perspective correction utilities for OCR images.*
        - Exports: LineQualityReport, MaskRectangleResult, calculate_target_dimensions, four_point_transform, correct_perspective_from_mask, ... (8 total)
        📄 `__init__.py`
        📄 `core.py` - Functions: calculate_target_dimensions, four_point_transform, correct_perspective_from_mask
        📄 `fitting.py` - Functions: fit_mask_rectangle
        📄 `geometry.py`
        📄 `quality_metrics.py`
        📄 `types.py` - Classes: LineQualityReport, MaskRectangleResult
        📄 `validation.py`
      📄 `__init__.py`
      📄 `api_usage_tracker.py` - Classes: APIUsageRecord, APIUsageStats, UpstageAPITracker - Functions: get_tracker
      📄 `background_normalization.py` - Functions: normalize_gray_world
      📄 `cache_manager.py` - Classes: CacheManager
      📄 `callbacks.py` - Functions: build_callbacks
      📄 `config.py` - Classes: ConfigParser
      📄 `config_utils.py` - Functions: is_config, ensure_dict, load_config
      📄 `config_validation.py` - Functions: validate_runtime, validate_config_paths
      📄 `convert_submission.py` - Functions: convert_json_to_csv, convert
      📄 `data_utils.py` - Functions: extract_metadata
      📄 `experiment_index.py` - Functions: get_next_experiment_index, get_current_experiment_index, reset_experiment_index
      📄 `experiment_name.py` - Functions: resolve_experiment_name, resolve_run_directory_experiment_name, find_run_dirs_for_exp_name
      📄 `geometry_utils.py` - Functions: calculate_inverse_transform, compute_padding_offsets, apply_padding_offset_to_polygons
      📄 `image_loading.py` - Functions: load_image_optimized, get_image_loader_info
      📄 `image_utils.py` - Functions: safe_get_image_size, load_pil_image, ensure_rgb
      📄 `logger_factory.py` - Functions: create_logger
      📄 `logging.py` - Classes: OCRLogger, DebugTools - Functions: log_experiment_start, log_experiment_end, create_experiment_logger
      📄 `ocr_utils.py` - Functions: draw_boxes
      📄 `orientation.py` - Functions: get_exif_orientation, orientation_requires_rotation, normalize_pil_image
      📄 `orientation_constants.py` - Classes: OrientationTransform - Functions: get_orientation_transform, get_inverse_orientation
      📄 `path_utils.py` - Classes: OCRPathConfig, OCRPathResolver - Functions: get_path_resolver, setup_project_paths, ensure_output_dirs
      📄 `polygon_utils.py` - Functions: ensure_polygon_array, filter_degenerate_polygons, validate_polygon_finite
      📄 `registry.py` - Classes: ComponentRegistry - Functions: get_registry
      📄 `sepia_enhancement.py` - Functions: enhance_sepia, enhance_clahe, enhance_sepia_clahe
      📄 `submission.py` - Classes: SubmissionWriter
      📄 `text_rendering.py` - Functions: get_korean_font, put_text_utf8, put_text_with_outline
      📄 `wandb_utils.py` - Functions: load_env_variables, generate_run_name, finalize_run
    📄 `__init__.py`
    📄 `experiment.py` - Classes: ExperimentMetadata, ExperimentRegistry - Functions: get_registry
    📄 `validation.py` - Classes: CacheConfig, ImageLoadingConfig, DatasetConfig - Functions: validate_predictions
  📁 **data/** - *📊 Data*
    📁 **datasets/** - *📊 Data* - *OCR datasets package.*
      - Exports: ValidatedOCRDataset, CraftCollateFN, DBCollateFN, DocumentPreprocessor, LensStylePreprocessorAlbumentations, ... (7 total)
      📁 **preprocessing/** - *Preprocessing submodule exposing modular document preprocessing components.*
        - Exports: A, ALBUMENTATIONS_AVAILABLE, AdvancedDetectionConfig, AdvancedDocumentDetector, AdvancedDocumentPreprocessor, ... (28 total)
        📁 **archive/**
          📁 **phase1_experimental_modules/**
            📄 `README.md`
        📄 `__init__.py`
        📄 `advanced_detector.py` - Classes: DetectionHypothesis, AdvancedDetectionConfig, AdvancedDocumentDetector
        📄 `advanced_noise_elimination.py` - Classes: NoiseReductionMethod, NoiseEliminationConfig, NoiseEliminationQualityMetrics - Functions: validate_noise_elimination_result
        📄 `advanced_preprocessor.py` - Classes: AdvancedPreprocessingConfig, AdvancedDocumentPreprocessor, OfficeLensPreprocessorAlbumentations - Functions: create_legacy_office_lens_preprocessor, create_high_accuracy_preprocessor
        📄 `background_removal.py` - Classes: BackgroundRemoval - Functions: create_background_removal_transform
        📄 `config.py` - Classes: EnhancementMethod, DocumentPreprocessorConfig
        📄 `contracts.py` - Classes: ImageInputContract, PreprocessingResultContract, DetectionResultContract - Functions: validate_image_input, validate_preprocessing_result, validate_image_input_with_fallback
        📄 `detector.py` - Classes: DocumentDetector
        📄 `document_flattening.py` - Classes: FlatteningMethod, FlatteningConfig, SurfaceNormals - Functions: flatten_crumpled_document
        📄 `enhanced_pipeline.py` - Classes: EnhancementStage, QualityThresholds, EnhancedPipelineConfig - Functions: create_office_lens_preprocessor, create_fast_preprocessor
        📄 `enhancement.py` - Classes: ImageEnhancer
        📄 `external.py`
        📄 `intelligent_brightness.py` - Classes: BrightnessMethod, BrightnessConfig, BrightnessQuality - Functions: create_brightness_adjuster
        📄 `metadata.py` - Classes: ImageShape, DocumentMetadata, PreprocessingState
        📄 `orientation.py` - Classes: OrientationCorrector
        📄 `padding.py` - Classes: PaddingCleanup
        📄 `perspective.py` - Classes: PerspectiveCorrector
        📄 `pipeline.py` - Classes: DocumentPreprocessor
        📄 `resize.py` - Classes: FinalResizer
        📄 `telemetry.py` - Classes: TelemetryEvent, PreprocessingTelemetry
        📄 `validators.py` - Classes: ImageValidator, ContractValidator, NumpyArray
      📄 `__init__.py` - Functions: get_datasets_by_cfg
      📄 `base.py` - Classes: ValidatedOCRDataset
      📄 `craft_collate_fn.py` - Classes: CraftCollateFN
      📄 `db_collate_fn.py` - Classes: DBCollateFN
      📄 `recognition_collate_fn.py` - Functions: recognition_collate_fn
      📄 `schemas.py`
      📄 `transforms.py` - Classes: ConditionalNormalize, ValidatedDBTransforms
    📁 **schemas/**
      📄 `storage.py` - Classes: BaseStorageItem, OCRStorageItem, KIEStorageItem
    📄 `charset.json`
  📁 **features/**
    📁 **detection/** - *Text Detection feature package.*
      📁 **models/** - *🤖 Models* - *Detection model components.*
        - Exports: CRAFT, DBNet, DBNetPP, CRAFTHead, DBHead, ... (11 total)
        📁 **architectures/** - *Detection architecture definitions.*
          - Exports: CRAFT, DBNet, DBNetPP
          📄 `__init__.py`
          📄 `craft.py` - Functions: register_craft_components
          📄 `dbnet.py` - Functions: register_dbnet_components
          📄 `dbnetpp.py` - Functions: register_dbnetpp_components
        📁 **decoders/** - *Detection decoder definitions.*
          - Exports: CRAFTDecoder, DBPPDecoder, FPNDecoder
          📄 `__init__.py`
          📄 `craft_decoder.py` - Classes: CraftDecoder
          📄 `dbpp_decoder.py` - Classes: DepthwiseSeparableConv, DBPPDecoder
          📄 `fpn_decoder.py` - Classes: FPNDecoder
        📁 **encoders/** - *Detection encoder definitions.*
          - Exports: CRAFTVGG
          📄 `__init__.py`
          📄 `craft_vgg.py` - Classes: CraftVGGEncoder
        📁 **heads/** - *Detection head definitions.*
          - Exports: CRAFTHead, DBHead
          📄 `__init__.py`
          📄 `craft_head.py` - Classes: CraftHead
          📄 `db_head.py` - Classes: DBHead
        📁 **postprocess/** - *Detection postprocessing utilities.*
          - Exports: CRAFTPostProcessor, DBPostProcessor
          📄 `__init__.py`
          📄 `craft_postprocess.py` - Classes: CraftPostProcessor
          📄 `db_postprocess.py` - Classes: DBPostProcessor
        📄 `__init__.py`
      📄 `__init__.py`
      📄 `interfaces.py` - Classes: DetectionHead, DetectionLoss
    📁 **kie/** - *Key Information Extraction (KIE) feature package.*
      📁 **data/** - *📊 Data* - *KIE data handling.*
        - Exports: KIEDataset
        📄 `__init__.py`
        📄 `dataset.py` - Classes: KIEDataset
      📁 **inference/** - *🔮 Inference*
        📁 **extraction/** - *Receipt data extraction module for OCR pipeline.*
          - Exports: LineItem, ReceiptData, ReceiptMetadata, ReceiptFieldExtractor, ExtractorConfig, ... (9 total)
          📄 `__init__.py`
          📄 `field_extractor.py` - Classes: ExtractorConfig, ReceiptFieldExtractor
          📄 `normalizers.py` - Functions: normalize_currency, normalize_date, normalize_time
          📄 `receipt_schema.py` - Classes: LineItem, ReceiptMetadata, ReceiptData
          📄 `vlm_extractor.py` - Classes: VLMExtractorConfig, VLMExtractor
      📁 **lightning/**
        📁 **callbacks/**
          📄 `__init__.py`
          📄 `kie_wandb_image_logging.py` - Classes: WandBKeyInformationExtractionImageLogger
        📄 `__init__.py`
      📁 **models/** - *🤖 Models* - *KIE model definitions.*
        - Exports: LayoutLMv3Wrapper, LiLTWrapper
        📄 `__init__.py`
        📄 `model.py` - Classes: LayoutLMv3Wrapper, LiLTWrapper
      📄 `__init__.py`
      📄 `trainer.py` - Classes: KIEDataPLModule, KIEPLModule
      📄 `validation.py` - Classes: KIEDataItem
    📁 **layout/** - *Layout detection feature for OCR pipeline.*
      - Exports: BoundingBox, LayoutResult, TextBlock, TextElement, TextLine, ... (8 total)
      📁 **inference/** - *🔮 Inference* - *Layout detection module for OCR pipeline.*
        - Exports: BoundingBox, TextElement, TextLine, TextBlock, LayoutResult, ... (7 total)
        📄 `__init__.py`
        📄 `contracts.py` - Classes: BoundingBox, TextElement, TextLine
        📄 `grouper.py` - Classes: LineGrouperConfig, LineGrouper - Functions: create_text_element
      📄 `README.md`
      📄 `__init__.py`
    📁 **recognition/**
      📁 **callbacks/**
        📄 `__init__.py`
        📄 `wandb_image_logging.py` - Classes: RecognitionWandbImageLogger
      📁 **data/** - *📊 Data* - *Recognition data module - tokenizers and datasets.*
        - Exports: KoreanOCRTokenizer, LMDBRecognitionDataset
        📄 `__init__.py`
        📄 `lmdb_dataset.py` - Classes: LMDBRecognitionDataset
        📄 `tokenizer.py` - Classes: KoreanOCRTokenizer
      📁 **inference/** - *🔮 Inference*
        📁 **backends/** - *Recognition backend implementations.*
          📄 `__init__.py`
          📄 `paddleocr_recognizer.py` - Classes: PaddleOCRRecognizer
        📄 `recognizer.py` - Classes: RecognizerBackend, RecognitionInput, RecognitionOutput
      📁 **models/** - *🤖 Models* - *Recognition-specific model components.*
        - Exports: PARSeq, PARSeqDecoder, PARSeqHead, register_parseq_components
        📄 `__init__.py`
        📄 `architecture.py` - Classes: PARSeq - Functions: register_parseq_components
        📄 `decoder.py` - Classes: PARSeqDecoder
        📄 `head.py` - Classes: PARSeqHead
  📁 **synthetic_data/** - *Modular synthetic data generation for OCR training.*
    - Exports: SyntheticDatasetGenerator, TextGenerator, BackgroundGenerator, TextRenderer, SyntheticImage, ... (9 total)
    📁 **generators/** - *Synthetic data generators for text, backgrounds, and rendering.*
      - Exports: TextGenerator, BackgroundGenerator, TextRenderer
      📄 `__init__.py`
      📄 `background.py` - Classes: BackgroundGenerator
      📄 `renderer.py` - Classes: TextRenderer
      📄 `text.py` - Classes: TextGenerator
    📄 `__init__.py`
    📄 `dataset.py` - Classes: SyntheticDatasetGenerator
    📄 `models.py` - Classes: TextRegion, SyntheticImage
    📄 `utils.py` - Functions: create_synthetic_dataset, augment_existing_dataset, setup_augmentation_pipeline
  📁 **validation/**
    📄 `models.py`
  📄 `__init__.py`
  📄 `experiment_registry.py` - Classes: ExperimentMetadata, ExperimentRegistry - Functions: get_registry

**Summary**: 64 directories, 231 files
