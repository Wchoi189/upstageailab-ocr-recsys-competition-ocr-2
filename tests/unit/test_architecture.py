from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from ocr.core.models.architecture import OCRModel


class TestOCRModel:
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "encoder": {},
                "decoder": {},
                "head": {},
                "loss": {},
                "optimizer": {},
            }
        )

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 3, 224, 224)  # batch_size=2, channels=3, height=224, width=224

    @patch("ocr.core.models.encoder.get_encoder_by_cfg")
    @patch("ocr.core.models.decoder.get_decoder_by_cfg")
    @patch("ocr.core.models.head.get_head_by_cfg")
    @patch("ocr.core.models.loss.get_loss_by_cfg")
    def test_model_initialization(self, mock_loss, mock_head, mock_decoder, mock_encoder, mock_config):
        """Test that OCRModel initializes correctly with mocked components."""
        # Setup mocks
        mock_encoder.return_value = Mock()
        mock_decoder.return_value = Mock()
        mock_head.return_value = Mock()
        mock_loss.return_value = Mock()

        # Initialize model
        model = OCRModel(mock_config)

        # Verify components are created
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert hasattr(model, "head")
        assert hasattr(model, "loss")
        assert model.cfg == mock_config

        # Verify factory functions were called
        mock_encoder.assert_called_once_with(mock_config.encoder)
        mock_decoder.assert_called_once_with(mock_config.decoder)
        mock_head.assert_called_once_with(mock_config.head)
        mock_loss.assert_called_once_with(mock_config.loss)

    def test_forward_pass_with_loss(self, mock_config, sample_input):
        """Test forward pass that includes loss calculation."""
        with (
            patch("ocr.core.models.encoder.get_encoder_by_cfg") as mock_enc,
            patch("ocr.core.models.decoder.get_decoder_by_cfg") as mock_dec,
            patch("ocr.core.models.head.get_head_by_cfg") as mock_head,
            patch("ocr.core.models.loss.get_loss_by_cfg") as mock_loss_func,
        ):
            # Setup component mocks
            mock_encoder = Mock()
            mock_decoder = Mock()
            mock_head_comp = Mock()
            mock_loss_comp = Mock()

            mock_enc.return_value = mock_encoder
            mock_dec.return_value = mock_decoder
            mock_head.return_value = mock_head_comp
            mock_loss_func.return_value = mock_loss_comp

            # Setup return values
            encoded_features = torch.randn(2, 64, 56, 56)
            decoded_features = torch.randn(2, 32, 112, 112)
            pred = {"maps": torch.randn(2, 1, 224, 224)}
            loss = 0.5
            loss_dict = {"total_loss": loss}

            mock_encoder.return_value = encoded_features
            mock_decoder.return_value = decoded_features
            mock_head_comp.return_value = pred
            mock_loss_comp.return_value = (loss, loss_dict)

            # Initialize and run model
            model = OCRModel(mock_config)
            result = model(sample_input, return_loss=True, gt_maps=torch.randn(2, 1, 224, 224))

            # Verify result structure
            assert "maps" in result
            assert "loss" in result
            assert "loss_dict" in result
            assert result["loss"] == loss
            assert result["loss_dict"] == loss_dict

    def test_forward_pass_without_loss(self, mock_config, sample_input):
        """Test forward pass without loss calculation."""
        with (
            patch("ocr.core.models.encoder.get_encoder_by_cfg") as mock_enc,
            patch("ocr.core.models.decoder.get_decoder_by_cfg") as mock_dec,
            patch("ocr.core.models.head.get_head_by_cfg") as mock_head,
            patch("ocr.core.models.loss.get_loss_by_cfg") as mock_loss_func,
        ):
            # Setup component mocks
            mock_encoder = Mock()
            mock_decoder = Mock()
            mock_head_comp = Mock()

            mock_enc.return_value = mock_encoder
            mock_dec.return_value = mock_decoder
            mock_head.return_value = mock_head_comp
            mock_loss_func.return_value = Mock()

            # Setup return values
            encoded_features = torch.randn(2, 64, 56, 56)
            decoded_features = torch.randn(2, 32, 112, 112)
            pred = {"maps": torch.randn(2, 1, 224, 224)}

            mock_encoder.return_value = encoded_features
            mock_decoder.return_value = decoded_features
            mock_head_comp.return_value = pred

            # Initialize and run model
            model = OCRModel(mock_config)
            result = model(sample_input, return_loss=False)

            # Verify result structure
            assert "maps" in result
            assert "loss" not in result
            assert "loss_dict" not in result

    @patch("ocr.core.models.architecture.instantiate")
    def test_get_optimizers_without_scheduler(self, mock_instantiate, mock_config):
        """Test optimizer creation without scheduler."""
        with (
            patch("ocr.core.models.encoder.get_encoder_by_cfg"),
            patch("ocr.core.models.decoder.get_decoder_by_cfg"),
            patch("ocr.core.models.head.get_head_by_cfg"),
            patch("ocr.core.models.loss.get_loss_by_cfg"),
        ):
            # Legacy test - get_optimizers() no longer exists in V5
            # Optimizer configuration is now handled by Lightning module only
            # See ocr.core.lightning.base.OCRPLModule.configure_optimizers()
            model = OCRModel(mock_config)
            
            # Verify model is optimizer-agnostic
            assert not hasattr(model, "get_optimizers")
            assert not hasattr(model, "_get_optimizers_impl")

    @patch("ocr.core.models.architecture.instantiate")
    def test_get_optimizers_with_scheduler(self, mock_instantiate, mock_config):
        """Test that models no longer handle optimizer/scheduler creation (V5)."""
        with (
            patch("ocr.core.models.encoder.get_encoder_by_cfg"),
            patch("ocr.core.models.decoder.get_decoder_by_cfg"),
            patch("ocr.core.models.head.get_head_by_cfg"),
            patch("ocr.core.models.loss.get_loss_by_cfg"),
        ):
            # V5 Standard: Models are optimizer-agnostic
            # Optimizer configuration moved to Lightning modules
            model = OCRModel(mock_config)
            
            # Verify methods don't exist
            assert not hasattr(model, "get_optimizers")
            assert not hasattr(model, "_get_optimizers_impl")

    def test_get_polygons_from_maps(self, mock_config):
        """Test polygon extraction from prediction maps."""
        with (
            patch("ocr.core.models.encoder.get_encoder_by_cfg"),
            patch("ocr.core.models.decoder.get_decoder_by_cfg"),
            patch("ocr.core.models.head.get_head_by_cfg") as mock_head,
            patch("ocr.core.models.loss.get_loss_by_cfg"),
        ):
            mock_head_comp = Mock()
            mock_head.return_value = mock_head_comp

            gt_maps = torch.randn(2, 1, 224, 224)
            pred_maps = torch.randn(2, 1, 224, 224)
            expected_polygons = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]])]

            mock_head_comp.get_polygons_from_maps.return_value = expected_polygons

            model = OCRModel(mock_config)
            result = model.get_polygons_from_maps(gt_maps, pred_maps)

            assert result == expected_polygons
            mock_head_comp.get_polygons_from_maps.assert_called_once_with(gt_maps, pred_maps)

    def test_model_initialization_with_registry(self, mock_config):
        with (
            patch("ocr.core.models.architecture.get_registry") as mock_get_registry,
            patch("ocr.core.models.decoder.get_decoder_by_cfg"),
            patch("ocr.core.models.encoder.get_encoder_by_cfg"),
            patch("ocr.core.models.head.get_head_by_cfg"),
            patch("ocr.core.models.loss.get_loss_by_cfg"),
        ):
            encoder = Mock()
            decoder = Mock()
            head = Mock()
            loss = Mock()

            registry_mock = Mock()
            registry_mock.create_architecture_components.return_value = {
                "encoder": encoder,
                "decoder": decoder,
                "head": head,
                "loss": loss,
            }
            mock_get_registry.return_value = registry_mock

            from omegaconf import OmegaConf

            OmegaConf.set_struct(mock_config, False)
            mock_config.architecture_name = "craft"
            mock_config.component_overrides = {}

            model = OCRModel(mock_config)

            assert model.encoder is encoder
            assert model.decoder is decoder
            assert model.head is head
            assert model.loss is loss
            from unittest.mock import ANY

            # When no component overrides are provided, only config keys are set
            registry_mock.create_architecture_components.assert_called_once_with(
                "craft",
                encoder_config=ANY,
                decoder_config=ANY,
                head_config=ANY,
                loss_config=ANY,
            )

    def test_component_override_with_decoder_name(self, mock_config):
        with patch("ocr.core.models.architecture.get_registry") as mock_get_registry:
            encoder = Mock(name="encoder")
            decoder = Mock(name="decoder")
            head = Mock(name="head")
            loss = Mock(name="loss")

            registry_mock = Mock()
            registry_mock.create_architecture_components.return_value = {
                "encoder": encoder,
                "decoder": decoder,
                "head": head,
                "loss": loss,
            }
            mock_get_registry.return_value = registry_mock

            from omegaconf import OmegaConf

            OmegaConf.set_struct(mock_config, False)
            mock_config.architecture_name = "dbnet"
            mock_config.component_overrides = {
                "decoder": {
                    "name": "fpn_decoder",
                    "params": {
                        "inner_channels": 128,
                        "out_channels": 128,
                    },
                }
            }

            OCRModel(mock_config)

            registry_mock.create_architecture_components.assert_called_once()
            _, kwargs = registry_mock.create_architecture_components.call_args
            assert kwargs["decoder_name"] == "fpn_decoder"
            assert kwargs["decoder_config"]["out_channels"] == 128

    def test_dbnet_architecture_passes_postprocess_config(self):
        with patch("ocr.core.models.architecture.get_registry") as mock_get_registry:
            encoder = Mock(name="encoder")
            decoder = Mock(name="decoder")
            head = Mock(name="head")
            loss = Mock(name="loss")

            def _assert_postprocess(architecture_name, **kwargs):
                assert architecture_name == "dbnet"
                head_config = kwargs["head_config"]
                assert "postprocess" in head_config.get("params", {})
                postprocess = head_config["params"]["postprocess"]
                assert postprocess == {
                    "thresh": 0.3,
                    "box_thresh": 0.4,
                    "max_candidates": 300,
                    "use_polygon": False,
                }
                return {
                    "encoder": encoder,
                    "decoder": decoder,
                    "head": head,
                    "loss": loss,
                }

            registry_mock = Mock()
            registry_mock.create_architecture_components.side_effect = _assert_postprocess
            mock_get_registry.return_value = registry_mock

            from omegaconf import OmegaConf

            cfg = OmegaConf.create(
                {
                    "architecture_name": "dbnet",
                    "component_overrides": {
                        "encoder": {},
                        "decoder": {},
                        "head": {
                            "params": {
                                "postprocess": {
                                    "thresh": 0.3,
                                    "box_thresh": 0.4,
                                    "max_candidates": 300,
                                    "use_polygon": False,
                                }
                            }
                        },
                        "loss": {},
                    },
                    "optimizer": {},
                }
            )

            OCRModel(cfg)

            registry_mock.create_architecture_components.assert_called_once()
    def test_prepare_component_configs_merge_order(self, mock_config):
        """Test strict merge order: Arch < Legacy < User."""
        from omegaconf import OmegaConf

        # 1. Architecture Default
        arch_conf = OmegaConf.create({
            "name": "arch_default",
            "component_overrides": {
               "encoder": {"name": "dummy_enc", "params": {"a": 1, "b": 1}}
            }
        })

        # 2. Legacy/Experiment (top-level)
        mock_config.encoder = {"params": {"b": 2, "c": 2}}

        # 3. User Override
        mock_config.component_overrides = {
            "encoder": {"params": {"c": 3, "d": 3}}
        }

        mock_config.architecture_name = arch_conf

        with patch("ocr.core.models.architecture.get_registry") as mock_reg:
             mock_reg.return_value.create_architecture_components.return_value = {
                 "encoder": Mock(), "decoder": Mock(), "head": Mock(), "loss": Mock()
             }
             with patch("ocr.core.models.encoder.get_encoder_by_cfg"), \
                  patch("ocr.core.models.decoder.get_decoder_by_cfg"), \
                  patch("ocr.core.models.head.get_head_by_cfg"), \
                  patch("ocr.core.models.loss.get_loss_by_cfg"):

                  model = OCRModel(mock_config)

                  # Inspect calls
                  _, kwargs = mock_reg.return_value.create_architecture_components.call_args
                  enc_conf = kwargs["encoder_config"]

                  # Expected:
                  # a=1 (Arch)
                  # b=2 (Legacy overrides Arch)
                  # c=3 (User overrides Legacy)
                  # d=3 (User)
                  assert enc_conf["a"] == 1
                  assert enc_conf["b"] == 2
                  assert enc_conf["c"] == 3
                  assert enc_conf["d"] == 3

    def test_filter_conflicts_legacy_vs_arch(self, mock_config):
        """Test BUG_003: Legacy config is filtered if it conflicts with Arch name."""
        from omegaconf import OmegaConf

        # Arch says encoder is "resnet"
        arch_conf = OmegaConf.create({
            "name": "my_arch",
            "encoder": {"name": "resnet", "params": {"depth": 50}}
        })

        # Legacy says encoder is "vgg" (Conflict!)
        mock_config.encoder = {"name": "vgg", "params": {"bn": False}}

        # User override says nothing about name, just params
        mock_config.component_overrides = {"encoder": {"params": {"pretrained": True}}}

        mock_config.architecture_name = arch_conf

        with patch("ocr.core.models.architecture.get_registry") as mock_reg:
             mock_reg.return_value.create_architecture_components.return_value = {
                 "encoder": Mock(), "decoder": Mock(), "head": Mock(), "loss": Mock()
             }
             with patch("ocr.core.models.encoder.get_encoder_by_cfg"), \
                  patch("ocr.core.models.decoder.get_decoder_by_cfg"), \
                  patch("ocr.core.models.head.get_head_by_cfg"), \
                  patch("ocr.core.models.loss.get_loss_by_cfg"):

                  OCRModel(mock_config)

                  _, kwargs = mock_reg.return_value.create_architecture_components.call_args

                  # Should be resnet (Legacy vgg filtered out)
                  assert kwargs["encoder_name"] == "resnet"
                  # Config should have depth=50 (Arch), pretrained=True (User)
                  # bn=False (Legacy) should be GONE because the whole legacy section was filtered
                  assert kwargs["encoder_config"]["depth"] == 50
                  assert kwargs["encoder_config"]["pretrained"] is True
                  assert "bn" not in kwargs["encoder_config"]
