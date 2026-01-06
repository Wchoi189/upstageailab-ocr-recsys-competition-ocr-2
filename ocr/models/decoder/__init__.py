from hydra.utils import instantiate

from .pan_decoder import PANDecoder  # noqa: F401
from .unet import UNetDecoder  # noqa: F401

# Backward compatibility
UNet = UNetDecoder


def get_decoder_by_cfg(config):
    return instantiate(config)
