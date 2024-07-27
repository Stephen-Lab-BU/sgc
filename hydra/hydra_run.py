from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_run(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    # parse params from cfg
    test = cfg.db.driver
    print(test)


    # check that data to fit model on exists
        # if it does not, generate data


if __name__ == "__main__":
    hydra_run() # pylint: disable=no-value-for-parameter