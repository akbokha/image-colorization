import os
import random
import numpy as np
import torch
from .options import ModelOptions


def main(options):

    # initialize random seed
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)

    if not os.path.exists(options.checkpoints_path):
        os.makedirs(options.checkpoints_path)

    # TODO: build model

    if options.mode == 0: # training mode

        args = vars(options)
        print('\n------------ Environment -------------')
        print('CUDA Available: {0}'.format(torch.cuda.is_available()))
        print('\n------------ Options -------------')
        with open(os.path.join(options.checkpoints_path, 'options.dat'), 'w') as f:
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
                f.write('%s: %s\n' % (str(k), str(v)))
        print('\n-------------- End ----------------\n')

        # TODO: train model

    elif options.mode == 1:
        # TODO: test model
        pass

if __name__ == "__main__":
    main(ModelOptions().parse())
