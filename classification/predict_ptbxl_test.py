from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from classification.tasks.ptbxl_test_prediction import run_ptbxl_test_prediction


@hydra.main(version_base=None, config_path="configs", config_name="predict_ptbxl_test")
def main(cfg: DictConfig) -> None:
    run_ptbxl_test_prediction(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
