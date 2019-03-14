import os
import matplotlib.pyplot as plt


def get_idx_to_label(dataset):
    """
    Get inverse list of index to label lookups
    """
    return {i[1]: i[0] for i in dataset.class_to_idx.items()}


def generate_eval_set(gpu_available, options, test_loader):
    class_idx_to_label = get_idx_to_label(test_loader.dataset)

    if options.eval_type == 'original':
        path = os.path.join('./eval/', options.eval_type)
        for i, (inputs, targets) in enumerate(test_loader):
            for j in range(inputs.shape[0]):
                img_data = inputs[j].detach().cpu().numpy()
                img_data = img_data.transpose((1, 2, 0))
                label = class_idx_to_label[targets[j].item()]
                file_name = 'img-{0:04d}.jpg'.format(i * test_loader.batch_size + j)
                img_path = os.path.join(path, label)

                if not os.path.exists(img_path):
                    os.makedirs(img_path)

                plt.imsave(arr=img_data, fname=os.path.join(img_path, file_name))

    elif options.eval_type == 'grayscale':
        print("generate grayscale eval set")
        # iterate data loader, convert to grayscale and stream images / labels to disk

    elif options.eval_type == 'colorize':
        print("generate grayscale eval set")
        # iterate data loader and colorization model from saved weights, colorise and stream images / labels to disk