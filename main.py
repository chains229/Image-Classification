import yaml
import argparse
from typing import Text
import transformers

from task.train import Training
from task.infer import Predicting

def main(config_path: Text) -> None:
    transformers.logging.set_verbosity_error()
    
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    
    # Train
    Training(config).main()
    
    # Inference
    Predicting(config).main()

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    
    main(args.config)