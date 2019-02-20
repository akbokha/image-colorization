import argparse
import os
import random


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class ModelOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(description='image-colorization')

        parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
        parser.add_argument('--experiment-name', type=str, default='experiment_001',
                            help='Experiment name (default: experiment_001)')
        parser.add_argument('--model-name', type=str, default='resnet', help='Model architecture (default: resnet)')
        parser.add_argument('--dataset-name', type=str, default='placeholder',
                            help='the input dataset to use [placeholder, places365] (default: placeholder)')
        parser.add_argument('--dataset-root-path', type=str, default='./data',
                            help='dataset root path (default: ./data)')
        parser.add_argument('--output-root-path', type=str, default='./output',
                            help='models, stats etc. are saved here (default: ./output)')
        parser.add_argument('--max_epochs', type=int, default='5', help='max number of epoch to train for')
        parser.add_argument('--batch-size', type=int, default='5', help='training batch size')
        parser.add_argument('--batch-output-frequency', type=int, default=1,
                            help='frequency with which to output batch stats')

        self._parser = parser

    def parse(self):
        opt = self._parser.parse_args()

        if opt.seed == 0:
            opt.seed = random.randint(0, 2 ** 31 - 1)

        opt.dataset_path = os.path.join(opt.dataset_root_path, opt.dataset_name)
        opt.experiment_output_path = os.path.join(opt.output_root_path, opt.experiment_name)

        return opt
