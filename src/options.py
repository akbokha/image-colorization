from __future__ import print_function
import os
import random
import argparse


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
        parser.add_argument('--model', type=str, default='test', help='Model name (default: test)')
        parser.add_argument('--mode', default=0, help='run mode [0: train, 1: test] (default: 0)')
        parser.add_argument('--dataset', type=str, default='cifar10',
                            help='the dataset to use [cifar10] (default: cifar10)')
        parser.add_argument('--dataset-path', type=str, default='./dataset', help='dataset path (default: ./dataset)')
        parser.add_argument('--checkpoints-path', type=str, default='./checkpoints',
                            help='models are saved here (default: ./checkpoints)')

        self._parser = parser

    def parse(self):
        opt = self._parser.parse_args()

        if opt.seed == 0:
            opt.seed = random.randint(0, 2 ** 31 - 1)

        if opt.dataset_path == './dataset':
            opt.dataset_path += ('/' + opt.dataset)

        if opt.checkpoints_path == './checkpoints':
            opt.checkpoints_path += ('/' + opt.dataset)

        return opt
