import hydra

from omegaconf import DictConfig

from train.trainer import Trainer


@hydra.main(version_base=None, config_path="./configs", config_name="main_config")
def main(config: DictConfig) -> None:
    print(config)
    trainer = Trainer(config)
    trainer.run_train()


if __name__ == "__main__":
    main()