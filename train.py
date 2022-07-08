import argparse

from model.trainer import Trainer
from tool.config import Cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='see example at ')
    parser.add_argument('--checkpoint', required=False, help='your checkpoint')
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)

    trainer = Trainer(config)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint, args.resume)

    trainer.train()


if __name__ == '__main__':
    main()
