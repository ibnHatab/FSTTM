
import argparse
from gpt_fsttm_server.config import parse_config

def parse_args():
    parser = argparse.ArgumentParser(description="Finite-State Turn-Taking Machine")
    parser.add_argument('--config', required=True,  help="Path of the server configuration file")
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    with open(args.config, 'r') as config_file:
        config = parse_config(config_file)

    print(config)

if __name__ == '__main__':
    import sys
    sys.argv.append('--config')
    sys.argv.append('config.sample.yaml')
    main()
