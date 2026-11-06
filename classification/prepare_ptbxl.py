from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from classification.data.ptbxl import prepare_ptbxl_dataset


@hydra.main(version_base=None, config_path="configs", config_name="prepare_ptbxl")
def main(cfg: DictConfig) -> None:
    prepare_ptbxl_dataset(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
