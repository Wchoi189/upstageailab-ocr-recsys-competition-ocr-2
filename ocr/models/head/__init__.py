from hydra.utils import instantiate


def get_head_by_cfg(config):
    return instantiate(config)
