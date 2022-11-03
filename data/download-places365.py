import os
import tarfile
import urllib.request
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

URL_PATH = 'http://data.csail.mit.edu/places/places365/'
FILENAME = 'places365standard_easyformat.tar'
URL = os.path.join(URL_PATH, FILENAME)
EXTRACTED_PATH = 'places365_standard'
OUTPUT_PATH = 'places365'
TRAIN_PATH = os.path.join(OUTPUT_PATH, 'train')
TEST_PATH = os.path.join(OUTPUT_PATH, 'test')
TRAIN_EXAMPLES_PER_CLASS = 1000
TEST_EXAMPLES_PER_CLASS = 100

print('Downloading TAR archive from {0}'.format(URL))

# Download dataset archive
download_url(URL, FILENAME)

print('Extracting files from {0}'.format(FILENAME))

# Extract files
with tarfile.open(FILENAME, 'r') as tar:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tar)

print('Restructuring dataset')

# Restructure and reduce size
os.rename(EXTRACTED_PATH, OUTPUT_PATH)

for idx, class_dir in enumerate(os.listdir(TRAIN_PATH)):
    if class_dir.startswith('.DS_'):
        continue

    print('Processing class {0}'.format(class_dir))

    train_class_path = os.path.join(TRAIN_PATH, class_dir)
    test_class_path = os.path.join(TEST_PATH, class_dir)

    if not os.path.exists(test_class_path):
        os.makedirs(test_class_path)

    for img_idx, img_file in enumerate(os.listdir(train_class_path)):
        img_train_path = os.path.join(train_class_path, img_file)
        if img_idx < TEST_EXAMPLES_PER_CLASS:
            # Move images to test directory
            img_test_path = os.path.join(test_class_path, img_file)
            os.rename(img_train_path, img_test_path)

        elif img_idx >= TEST_EXAMPLES_PER_CLASS + TRAIN_EXAMPLES_PER_CLASS:
            # Delete unnecessary images
            os.remove(img_train_path)



