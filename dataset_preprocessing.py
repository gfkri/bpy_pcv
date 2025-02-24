import logging
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import rootutils

log = logging.getLogger(__name__)


#######################################################################################################################
@hydra.main(version_base=None, config_path="conf", config_name="data_preparation_kitti")
def preprocess_data(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg)) 
    
    log.info("Instantiating preprocessor <%s>", cfg.preprocessor._target_)
    preprocessor = hydra.utils.instantiate(cfg.preprocessor)
    
    log.info("Running preprocessor")
    preprocessor.run(cfg)
    

#######################################################################################################################
if __name__ == '__main__':
    preprocess_data()