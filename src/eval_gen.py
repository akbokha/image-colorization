import os
import matplotlib.pyplot as plt


def get_idx_to_label(dataset):
    """
    Get inverse list of index to label lookups
    """
    return {i[1]: i[0] for i in dataset.class_to_idx.items()}


def save_image(img_data, root_path, label, file_name):
    """
    Save transformed image to disk
    """
    img_path = os.path.join(root_path, label)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    if len(img_data.shape) == 2:
        plt.imsave(arr=img_data, fname=os.path.join(img_path, file_name), cmap='gray')
    else:
        plt.imsave(arr=img_data, fname=os.path.join(img_path, file_name))


def generate_eval_set(gpu_available, options, test_loader):
    """
    Create images from test set for evaluation
    """
    class_idx_to_label = get_idx_to_label(test_loader.dataset)

    if options.eval_type == 'original':
        path = os.path.join('./eval/', options.eval_type)

        for i, (imgs_grayscale, imgs_ab, imgs_original, targets) in enumerate(test_loader):
            for j in range(imgs_original.shape[0]):
                img_data = imgs_original[j].detach().cpu().numpy()
                label = class_idx_to_label[targets[j].item()]
                file_name = 'img-{0:04d}.jpg'.format(i * test_loader.batch_size + j)
                save_image(img_data, path, label, file_name)

    elif options.eval_type == 'grayscale':
        path = os.path.join('./eval/', options.eval_type)

        for i, (imgs_grayscale, imgs_ab, imgs_original, targets) in enumerate(test_loader):
            for j in range(imgs_grayscale.shape[0]):
                img_data = imgs_grayscale[j].detach().cpu().squeeze().numpy()
                label = class_idx_to_label[targets[j].item()]
                file_name = 'img-{0:04d}.jpg'.format(i * test_loader.batch_size + j)
                save_image(img_data, path, label, file_name)

    elif options.eval_type == 'colorize':
        print("generate grayscale eval set")
        # iterate data loader and colorization model from saved weights, colorise and stream images / labels to disk