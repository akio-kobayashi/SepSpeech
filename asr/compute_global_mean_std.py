import generator
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')


def main(csv_path:str, config:dict):
    generator.compute_global_mean_std(csv_path, config['analysis'])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    args=parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)
    
    main(args.csv_path, config)