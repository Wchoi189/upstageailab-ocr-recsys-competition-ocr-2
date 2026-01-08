"""Unit tests for model manager."""


class TestModelManagerInit:
    """Tests for ModelManager initialization."""

    def test_init_with_default_device(self):
        """Test manager initialization with auto-detected device."""
        from ocr.core.inference.model_manager import ModelManager

        manager = ModelManager()
        assert manager.device in ("cuda", "cpu")
        assert manager.model is None
        assert manager.config is None
        assert manager._current_checkpoint_path is None

    def test_init_with_explicit_device(self):
        """Test manager initialization with explicit device."""
        from ocr.core.inference.model_manager import ModelManager

        manager = ModelManager(device="cpu")
        assert manager.device == "cpu"

    def test_init_with_cuda_device(self):
        """Test manager initialization with CUDA device."""
        from ocr.core.inference.model_manager import ModelManager

        manager = ModelManager(device="cuda")
        assert manager.device == "cuda"


class TestModelManagerState:
    """Tests for ModelManager state queries."""

    def test_is_loaded_initially_false(self):
        """Test that manager starts with no model loaded."""
        from ocr.core.inference.model_manager import ModelManager

        manager = ModelManager()
        assert manager.is_loaded() is False

    def test_get_current_checkpoint_initially_none(self):
        """Test that initially no checkpoint is loaded."""
        from ocr.core.inference.model_manager import ModelManager

        manager = ModelManager()
        assert manager.get_current_checkpoint() is None

    def test_get_config_bundle_initially_none(self):
        """Test that initially no config bundle exists."""
        from ocr.core.inference.model_manager import ModelManager

        manager = ModelManager()
        assert manager.get_config_bundle() is None


class TestModelManagerCleanup:
    """Tests for ModelManager cleanup."""

    def test_cleanup_when_no_model(self):
        """Test cleanup works when no model is loaded."""
        from ocr.core.inference.model_manager import ModelManager

        manager = ModelManager()
        manager.cleanup()  # Should not raise

        assert manager.model is None
        assert manager.config is None
        assert manager._config_bundle is None
        assert manager._current_checkpoint_path is None

    def test_context_manager_cleanup(self):
        """Test context manager calls cleanup on exit."""
        from ocr.core.inference.model_manager import ModelManager

        with ModelManager() as manager:
            assert manager is not None

        # After exit, cleanup should have been called
        assert manager.model is None


class TestModelManagerLoadModel:
    """Tests for ModelManager.load_model method."""

    def test_load_model_without_ocr_modules(self, monkeypatch):
        """Test that load fails gracefully when OCR modules unavailable."""
        # Mock OCR_MODULES_AVAILABLE to False
        import ocr.core.inference.model_manager as mm_module
        from ocr.core.inference.model_manager import ModelManager

        monkeypatch.setattr(mm_module, "OCR_MODULES_AVAILABLE", False)

        manager = ModelManager()
        result = manager.load_model("fake_checkpoint.pth")

        assert result is False
        assert manager.is_loaded() is False

    def test_load_model_with_nonexistent_checkpoint(self):
        """Test loading from nonexistent checkpoint."""
        from ocr.core.inference.model_manager import ModelManager

        manager = ModelManager()
        result = manager.load_model("/nonexistent/checkpoint.pth")

        # Should fail (no config found)
        assert result is False
        assert manager.is_loaded() is False


class TestModelManagerExtractModelConfig:
    """Tests for _extract_model_config static method."""

    def test_extract_model_config_with_model_section(self):
        """Test extracting config when 'model' section exists."""
        from ocr.core.inference.config_loader import ModelConfigBundle, PostprocessSettings, PreprocessSettings
        from ocr.core.inference.model_manager import ModelManager

        # Create mock config with 'model' attribute
        class MockConfig:
            def __init__(self):
                self.model = "model_config_object"

        raw_config = MockConfig()
        bundle = ModelConfigBundle(
            raw_config=raw_config,
            preprocess=PreprocessSettings(image_size=640, normalization={"mean": [0.5], "std": [0.5]}),
            postprocess=PostprocessSettings(
                binarization_thresh=0.3,
                box_thresh=0.7,
                max_candidates=1000,
                min_detection_size=3,
            ),
        )

        result = ModelManager._extract_model_config(bundle, "test_config.yaml")

        assert result == "model_config_object"

    def test_extract_model_config_without_model_section(self):
        """Test extracting config when no 'model' section (Hydra-style)."""
        from ocr.core.inference.config_loader import ModelConfigBundle, PostprocessSettings, PreprocessSettings
        from ocr.core.inference.model_manager import ModelManager

        # Create mock config with model attributes at root
        class MockConfig:
            def __init__(self):
                self.architecture = "resnet"
                self.encoder = {}
                self.decoder = {}

        raw_config = MockConfig()
        bundle = ModelConfigBundle(
            raw_config=raw_config,
            preprocess=PreprocessSettings(image_size=640, normalization={"mean": [0.5], "std": [0.5]}),
            postprocess=PostprocessSettings(
                binarization_thresh=0.3,
                box_thresh=0.7,
                max_candidates=1000,
                min_detection_size=3,
            ),
        )

        result = ModelManager._extract_model_config(bundle, "test_config.yaml")

        assert result is raw_config

    def test_extract_model_config_invalid(self):
        """Test extracting config fails when neither format is present."""
        from ocr.core.inference.config_loader import ModelConfigBundle, PostprocessSettings, PreprocessSettings
        from ocr.core.inference.model_manager import ModelManager

        # Create mock config with no model-related attributes
        class MockConfig:
            def __init__(self):
                self.some_other_attr = "value"

        raw_config = MockConfig()
        bundle = ModelConfigBundle(
            raw_config=raw_config,
            preprocess=PreprocessSettings(image_size=640, normalization={"mean": [0.5], "std": [0.5]}),
            postprocess=PostprocessSettings(
                binarization_thresh=0.3,
                box_thresh=0.7,
                max_candidates=1000,
                min_detection_size=3,
            ),
        )

        result = ModelManager._extract_model_config(bundle, "test_config.yaml")

        assert result is None
