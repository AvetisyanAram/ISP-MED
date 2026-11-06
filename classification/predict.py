from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from classification.tasks.prediction import run_prediction


@hydra.main(version_base=None, config_path="configs", config_name="predict")
def main(cfg: DictConfig) -> None:
    run_prediction(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
