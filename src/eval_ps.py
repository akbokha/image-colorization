from .utils import *
from .ps.dist_model import DistModel
from .ps import ps_util

def evaluate_ps(gpu_available, options):

    model_path = os.path.join(options.model_path, 'alex.pth')
    model = DistModel()
    model.initialize(model='net-lin', net='alex', model_path=model_path, use_gpu=gpu_available)

    dist_stats = AverageMeter()

    original_path = os.path.join(options.eval_root_path, 'original/test')
    if options.eval_type == 'colorized':
        compare_path = os.path.join(options.eval_root_path, options.full_model_name, 'test')
    else:
        compare_path = os.path.join(options.eval_root_path, options.eval_type, 'test')

    for class_dir in os.listdir(original_path):
        if not os.path.isdir(os.path.join(original_path, class_dir)):
            continue

        original_class_path = os.path.join(original_path, class_dir)
        compare_class_path = os.path.join(compare_path, class_dir)

        class_sum_distances = 0
        class_count = 0
        for img_file in os.listdir(original_class_path):
            if not os.path.isfile(os.path.join(original_path, img_file)) and img_file[-3:] != 'jpg':
                continue

            img_original = ps_util.im2tensor(
                ps_util.load_image(os.path.join(original_class_path, img_file)))
            img_compare = ps_util.im2tensor(
                ps_util.load_image(os.path.join(compare_class_path, img_file)))

            distance = model.forward(img_original, img_compare)[0]
            dist_stats.update(distance, 1)

        print_ts('Folder: {0}\tavg_dist {1:.3f}\tse_dist +/-{2:.3f}'.format(class_dir, dist_stats.avg, dist_stats.se))

    output_path = options.experiment_output_path
    epoch_stats = { 'avg_dist': [dist_stats.avg], 'se_dist': [dist_stats.se]}
    save_stats(output_path, 'ps_distances.csv', epoch_stats, 1)
