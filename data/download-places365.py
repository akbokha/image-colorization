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
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

URL_PATH = 'http://data.csail.mit.edu/places/places365/'
#FILENAME = 'places365standard_easyformat.tar'
FILENAME = 'val_256.tar'
URL = os.path.join(URL_PATH, FILENAME)
LOCAL_ROOT_PATH = 'places365/'
LOCAL_FILE_PATH = os.path.join(LOCAL_ROOT_PATH, FILENAME)

print('Downloading '.format(URL))

# Download dataset archive
#if not os.path.exists(LOCAL_ROOT_PATH):
#    os.makedirs(LOCAL_ROOT_PATH)
#download_url(URL, LOCAL_FILE_PATH)

print('Download completed')

# Extract files
with tarfile.open(LOCAL_FILE_PATH, 'r') as tar:
    tar.extractall(path=LOCAL_ROOT_PATH)

print('Extraction completed')

# TODO: Restructure and reduce size
