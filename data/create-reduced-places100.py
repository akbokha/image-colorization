import os
import numpy as np
import shutil

TRAIN_EXAMPLES_PER_CLASS = 500
VAL_EXAMPLES_PER_CLASS = 50
TEST_EXAMPLES_PER_CLASS = 50

INPUT_PATH = 'places365'
TRAIN_INPUT_PATH = os.path.join(INPUT_PATH, 'train')
VAL_INPUT_PATH = os.path.join(INPUT_PATH, 'val')
TEST_INPUT_PATH = os.path.join(INPUT_PATH, 'test')

OUTPUT_PATH = 'places100'
TRAIN_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'train')
VAL_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'val')
TEST_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'test')

all_classes = []
for idx, class_dir in enumerate(os.listdir(TRAIN_INPUT_PATH)):

    if class_dir.startswith('.DS_'):
        continue

    all_classes.append(class_dir)

np.random.seed(1)
class_arr = np.array(all_classes)
np.random.shuffle(class_arr)
reduced_classes = class_arr[0:100].tolist()

for cls_idx, class_name in enumerate(reduced_classes):

    print('processing: {}'.format(class_name))

    train_class_output_path = os.path.join(TRAIN_OUTPUT_PATH, class_name)
    if not os.path.exists(train_class_output_path):
        os.makedirs(train_class_output_path)

    for idx, file_name in enumerate(os.listdir(os.path.join(TRAIN_INPUT_PATH, class_name))):
        if idx < TRAIN_EXAMPLES_PER_CLASS:
            shutil.copyfile(os.path.join(TRAIN_INPUT_PATH, class_name, file_name), os.path.join(train_class_output_path, file_name))

    val_class_output_path = os.path.join(VAL_OUTPUT_PATH, class_name)
    if not os.path.exists(val_class_output_path):
        os.makedirs(val_class_output_path)

    for idx, file_name in enumerate(os.listdir(os.path.join(VAL_INPUT_PATH, class_name))):
        if idx < VAL_EXAMPLES_PER_CLASS:
            shutil.copyfile(os.path.join(VAL_INPUT_PATH, class_name, file_name), os.path.join(val_class_output_path, file_name))

    test_class_output_path = os.path.join(TEST_OUTPUT_PATH, class_name)
    if not os.path.exists(test_class_output_path):
        os.makedirs(test_class_output_path)

    for idx, file_name in enumerate(os.listdir(os.path.join(TEST_INPUT_PATH, class_name))):
        if idx < TEST_EXAMPLES_PER_CLASS:
            shutil.copyfile(os.path.join(TEST_INPUT_PATH, class_name, file_name), os.path.join(test_class_output_path, file_name))