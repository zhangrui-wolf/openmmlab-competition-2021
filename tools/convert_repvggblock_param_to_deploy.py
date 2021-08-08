import argparse


def convert_repvggblock_param(config_path,
                              checkpoint_path,
                              save_path,
                              device='cuda'):
    pass


def main():
    parser = argparse.ArgumentParser(
        description='Convert the parameters of the repvgg block '
        'from training mode to deployment mode.')
    parser.add_argument(
        'config_path',
        help='The path to the configuration file of the network '
        'containing the repvgg block.')
    parser.add_argument(
        'checkpoint_path',
        help='The path to the checkpoint file corresponding to the model.')
    parser.add_argument(
        'save_path',
        help='The path where the converted checkpoint file is stored.')
    parser.add_argument(
        '--device',
        default='cuda',
        help='The device to which the model is loaded.')
    args = parser.parse_args()
    convert_repvggblock_param(
        args.config_path,
        args.checkpoint_path,
        args.save_path,
        device=args.device)


if __name__ == '__main__':
    main()
