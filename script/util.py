import os
import urllib.request as urllib
import tarfile
import numpy as np


def download(url, path, overwrite=False):
    if os.path.isfile(path) and not overwrite:
        print('File {} existed, skip.'.format(path))
        return
    print('Downloading from url {} to {}'.format(url, path))
    try:
        urllib.request.urlretrieve(url, path)
    except:
        urllib.urlretrieve(url, path)


def extract_all(archive_path, extract_root_path):
    # extract all files from archive
    with tarfile.open(archive_path) as tar:
        tar.extractall(path=extract_root_path)

    # return extract root directory path
    return get_extract_dir_path(archive_path, extract_root_path)


def get_extract_dir_path(archive_path, extract_root_path):
    zip_file_name = os.path.basename(archive_path)
    name_pair = os.path.splitext(zip_file_name)
    name_without_ext = name_pair[0]
    if len(name_pair[1]) != 0:
        name_without_ext = os.path.splitext(name_without_ext)[0]
    return os.path.join(extract_root_path, name_without_ext)


def random_tensor(shape, scale=1, shift=0, seed=100):
    """
    指定したshapeに合わせて、ランダムなテンソルを作成する

    :param shape: テンソルの形状を示すリストを指定(ex. (1,2,3,4)など)
    :param scale: テンソル内の値の最大値を指定
    :param shift: テンソル内の値のシフト数を指定
    :param seed:
    :return:
    """
    np.random.seed(seed)

    # calculate item count
    count = 1
    for value in shape:
        count *= value

    # create random tensor
    tmp = np.random.random(count) * scale + shift
    return np.reshape(tmp, shape)


def draw_image(source, image, x, y):
    """
        Places "image" with alpha channel on "source" at x, y
        :param source:
        :type source: numpy.ndarray
        :param image:
        :type image: numpy.ndarray (with alpha channel)
        :param x:
        :type x: int
        :param y:
        :type y: int
        :rtype: numpy.ndarray
    """
    h, w = image.shape[:2]

    max_x, max_y = x + w, y + h
    alpha = image[:, :, 3] / 255.0
    for c in range(0, 3):
        color = image[:, :, c] * (alpha)
        beta = source[y:max_y, x:max_x, c] * (1.0 - alpha)
        source[y:max_y, x:max_x, c] = color + beta
    return source
