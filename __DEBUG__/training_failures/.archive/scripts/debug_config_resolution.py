import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base="1.3", config_path="../configs", config_name="main")
def main(cfg: DictConfig):
    print("--- Config Debug ---")
    print(f"Keys: {cfg.keys()}")
    # print(OmegaConf.to_yaml(cfg)) # Can be verbose, let's stick to keys first or structural summary

    if "domain" in cfg:
        print(f"cfg.domain type: {type(cfg.domain)}")
        print(f"cfg.domain content: {cfg.domain}")
        if isinstance(cfg.domain, DictConfig):
             print(f"cfg.domain.task: {cfg.domain.get('task', 'MISSING')}")
    else:
        print("cfg.domain is MISSING")

    # Check if task is at root
    print(f"cfg.task (root): {cfg.get('task', 'MISSING')}")

    # Check if 'recognition' key exists at root (maybe package issue?)
    if "recognition" in cfg:
        print("Found 'recognition' at root!")

    if "_group_" in cfg:
        print(f"Found '_group_' key! content: {cfg['_group_']}")

if __name__ == "__main__":
    main()
