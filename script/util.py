import os
import urllib.request as urllib
import tarfile
import numpy as np
from object_detection.utils import visualization_utils
from PIL import Image
from coco import coco
import psutil


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


def recreate_images_with_bounding_boxes(inp_files, input_tensor, res):
    boxes, classes, scores, num_det = res

    for i, fname in enumerate(inp_files):
        n_obj = int(num_det[i])
        image_array = input_tensor[i]

        print("{} - found objects:".format(fname))
        target_boxes = []
        for j in range(n_obj):
            # check score
            cl_id = int(classes[i][j])
            label = coco.IMAGE_CLASSES[cl_id]
            score = scores[i][j]
            if score < 0.5:
                continue

            # print each data
            box = boxes[i][j]
            print("  ", cl_id, label, score, box)
            target_boxes.append(box)

        # recreate image with bounding boxes
        visualization_utils.draw_bounding_boxes_on_image_array(image_array, np.array(target_boxes))
        out_file_name = os.path.splitext(fname)[0] + "_with_boxes.png"
        Image.fromarray(image_array).save(out_file_name)


def print_mem_usage():
    process = psutil.Process(os.getpid())
    print("Memory RSS: {:,}".format(process.memory_info().rss))


def get_ndarray_from_image(img_files, out_size):
    """
    get numpy.array from image with resizing.

    :param img_files : list
        path list of image file
    :param out_size : tuple
        out size tuple of image array.
        ex) (300, 300)
    :return: numpy.array
    """

    res = []
    for f in img_files:
        img = np.array(Image.open(f).resize(out_size))
        res.append(img)
    return np.array(res)
