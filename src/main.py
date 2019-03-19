import random
import sys

from .dataloaders import *
from .options import ModelOptions
from .utils import *
from .colorizer import train_colorizer
from .classifier import train_classifier
from .eval_gen import generate_eval_set
from .eval_si import evaluate_si
from .eval_ps import evaluate_ps
from .eval_mse import evaluate_mse

task_names = ['colorizer', 'classifier', 'eval-gen', 'eval-si', 'eval-ps', 'eval-mse']
dataset_names = ['placeholder', 'cifar10', 'places100', 'places205', 'places365']
dataset_label_counts = {
    'placeholder': 2,
    'places100': 100,
    'places365': 365
}
colorizer_model_names = ['resnet', 'unet32', 'cgan']


def main(options):
    # initialize random seed
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)

    # Determine if GPU is available
    gpu_available = torch.cuda.is_available()

    # Create experiment output directory
    if not os.path.exists(options.experiment_output_path):
        os.makedirs(options.experiment_output_path)

    args = vars(options)
    print('\n------------ Environment -------------')
    print('GPU Available: {0}'.format(gpu_available))
    print('\n------------ Options -------------')
    with open(os.path.join(options.experiment_output_path, 'options.dat'), 'w') as f:
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
            f.write('%s: %s\n' % (str(k), str(v)))

    # Check if specified dataset is one that is supported by experimentation framework
    if options.dataset_name not in dataset_names:
        print('{} is not a valid dataset. The supported datasets are: {}'.format(options.dataset_name, dataset_names))
        clean_and_exit(options)

    # Check if specified task is one that is supported by experimentation framework
    if options.task not in task_names:
        print('{} is not a valid task. The supported tasks are: {}'.format(options.task, task_names))
        clean_and_exit(options)

    if options.task == 'colorizer':

        # Create data loaders
        if options.dataset_name == 'placeholder':
            train_loader, val_loader = get_places_loaders(
                options.dataset_path, options.train_batch_size, options.val_batch_size, options.use_dataset_archive)

        elif options.dataset_name == 'cifar10':
            train_loader, val_loader = get_cifar10_loaders(
                options.dataset_path, options.train_batch_size, options.val_batch_size)

        elif options.dataset_name == 'places100':
            train_loader, val_loader = get_places_loaders(
                options.dataset_path, options.train_batch_size, options.val_batch_size, options.use_dataset_archive)

        elif options.dataset_name == 'places205':
            train_loader, val_loader = get_places205_loaders(
                options.dataset_path, options.train_batch_size, options.val_batch_size)

        elif options.dataset_name == 'places365':
            train_loader, val_loader = get_places_loaders(
                options.dataset_path, options.train_batch_size, options.val_batch_size, options.use_dataset_archive)

        # Check if specified model is one that is supported by experimentation framework
        if options.model_name not in colorizer_model_names:
            print('{} is not a valid model. The supported models are: {}'.format(
                options.model_name, colorizer_model_names))
            clean_and_exit(options)

        train_colorizer(gpu_available, options, train_loader, val_loader)

    elif options.task == 'classifier':

        if options.dataset_name in ['placeholder', 'places100', 'places365']:
            train_loader, val_loader = get_places_loaders(
                options.dataset_path, options.train_batch_size, options.val_batch_size, options.use_dataset_archive,
                for_classification=True)
            options.dataset_num_classes = dataset_label_counts[options.dataset_name]

        else:
            print("{} is not a valid dataset for classifier task".format(options.dataset_name))
            clean_and_exit(options)

        train_classifier(gpu_available, options, train_loader, val_loader)

    elif options.task == 'eval-gen':

        if options.dataset_name in ['placeholder', 'places100']:
            test_loader = get_places_test_loader(
                options.dataset_path, options.val_batch_size, options.use_dataset_archive)
        else:
            print("{} is not a valid dataset for eval-gen task".format(options.dataset_name))
            clean_and_exit(options)

        generate_eval_set(gpu_available, options, test_loader, False, resize=True)


    elif options.task == 'eval-si':
        if options.eval_type == 'colorized':
            dataset_path = os.path.join(options.eval_root_path, options.model_name)
        else:
            dataset_path = os.path.join(options.eval_root_path, options.eval_type)

        eval_loader = get_places_test_loader(
            dataset_path, options.val_batch_size, False, resize=False, for_classification=True)

        evaluate_si(gpu_available, options, eval_loader)

    elif options.task == 'eval-ps':

        evaluate_ps(gpu_available, options)

    elif options.task == 'eval-mse':

        original_dataset_path = os.path.join(options.eval_root_path, 'original')
        if options.eval_type == 'colorized':
            eval_dataset_path = os.path.join(options.eval_root_path, options.model_name)
        else:
            eval_dataset_path = os.path.join(options.eval_root_path, options.eval_type)

        original_loader = get_places_test_loader(
            original_dataset_path, options.val_batch_size, False, resize=False)
        eval_loader = get_places_test_loader(
            eval_dataset_path, options.val_batch_size, False, resize=False)

        evaluate_mse(gpu_available, options, original_loader, eval_loader)



def clean_and_exit(options):
    os.rmdir(options.experiment_output_path)
    sys.exit(1)


if __name__ == "__main__":
    main(ModelOptions().parse())
