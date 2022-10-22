import os
import urllib.request as urllib
import tarfile
import numpy as np
from object_detection.utils import visualization_utils
from PIL import Image
from coco import coco
import psutil
import cv2


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
            
        
        safe_extract(tar, path=extract_root_path)

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


def get_ndarray_from_imagefiles(img_files, out_size, transpose_tuple=None):
    """
    get numpy.array from image with resizing.

    :param img_files : list
        path list of image file
    :param out_size : tuple
        out size tuple of image array.
        ex) (300, 300)
    :param transpose_tuple : tuple
        transpose if needed
    :return: numpy.array
    """

    res = []
    for f in img_files:
        img = np.array(Image.open(f).resize(out_size))
        img = tranpose_if_needed(img, transpose_tuple)
        res.append(img)
    return np.array(res)


def get_input_data(model_define, input_tensor):
    if "input_tensor_name" in model_define.keys():
        input_data = {model_define["input_tensor_name"]: input_tensor}
    else:
        input_data = input_tensor
    return input_data


# TODO : delete this function
def open_and_norm_image(cv2_img, input_size, transpose_tuple=None):
    img = __resize_and_norm_image(cv2_img, input_size, transpose_tuple)
    img = np.expand_dims(img, axis=0)
    return img


def open_and_norm_images(cv2_images, input_size, transpose_tuple=None):
    images = []
    for cv2_img in cv2_images:
        img = __resize_and_norm_image(cv2_img, input_size, transpose_tuple)
        images.append(img)
    return np.array(images)


def __resize_and_norm_image(cv2_img, input_size, transpose_tuple=None):
    img = cv2.resize(cv2_img, input_size)
    img = img[:, :, (2, 1, 0)].astype(np.float32)
    img -= np.array([123, 117, 104])

    # transpose if needed
    return tranpose_if_needed(img, transpose_tuple)


def tranpose_if_needed(image_array, transpose_tuple=None):
    # need to transpose if model type is MXNet
    if transpose_tuple is not None:
        image_array = np.transpose(np.array(image_array), transpose_tuple)
    return image_array
