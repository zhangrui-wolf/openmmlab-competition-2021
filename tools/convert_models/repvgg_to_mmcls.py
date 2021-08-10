import argparse
from collections import OrderedDict
from pathlib import Path

import torch


def convert(src, dst):
    print('Converting...')
    blobs = torch.load(src, map_location='cpu')
    converted_state_dict = OrderedDict()

    for key in blobs:
        splited_key = key.split('.')
        splited_key = ['norm' if i == 'bn' else i for i in splited_key]
        splited_key = [
            'branch_identity' if i == 'rbr_identity' else i
            for i in splited_key
        ]
        splited_key = [
            'branch_1x1' if i == 'rbr_1x1' else i for i in splited_key
        ]
        splited_key = [
            'branch_3x3' if i == 'rbr_dense' else i for i in splited_key
        ]
        splited_key = [
            'backbone.stage_' + i[5] if i[:5] == 'stage' else i
            for i in splited_key
        ]
        splited_key = ['head.fc' if i == 'linear' else i for i in splited_key]
        new_key = '.'.join(splited_key)
        converted_state_dict[new_key] = blobs[key]

    torch.save(converted_state_dict, dst)
    print('Done!')


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    dst = Path(args.dst)
    if dst.suffix != '.pth':
        print('The path should contain the name of the pth format file.')
        exit()
    dst.parent.mkdir(parents=True, exist_ok=True)

    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
