from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from classification.tasks.training import run_training


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    run_training(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
