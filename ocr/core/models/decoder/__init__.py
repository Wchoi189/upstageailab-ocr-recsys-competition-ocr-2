from hydra.utils import instantiate




def get_decoder_by_cfg(config):
    return instantiate(config)
