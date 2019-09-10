import dlr
import psutil
import os
from model_loader import TfModelZooLoader, TfModelZooType
import util
from PIL import Image
import numpy as np
from coco import coco

from object_detection.utils import visualization_utils


def mem_usage():
    process = psutil.Process(os.getpid())
    print("Memory RSS: {:,}".format(process.memory_info().rss))


def get_random_input():
    input_tensor = util.random_tensor((1, 300, 300, 3))
    return input_tensor


def get_input(img_files, out_size):
    res = []
    for f in img_files:
        img = np.array(Image.open(f).resize(out_size))
        res.append(img)
    return np.array(res)


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


def main():
    # set model type
    model_type = TfModelZooType.SSD_MOBILE_NET_V2_COCO

    # load model data
    model_root_path = "model"
    loader = TfModelZooLoader(model_root_path, model_type.value["url"])
    loader.setup()
    model_info = loader.get_model()
    model_path = model_info.model_path_map["model_file"]

    # create Deep Learning Runtime
    m = dlr.DLRModel(model_path)

    # get input data
    input_files = ["data/dogs.jpg"]
    image_size = (300, 300)
    input_tensor = get_input(input_files, image_size)
    input_tensor_name = model_type.value["input_tensor_name"]

    # run inference
    res = m.run({input_tensor_name: input_tensor})

    # show inference result and recreate image files
    recreate_images_with_bounding_boxes(input_files, input_tensor, res)


if __name__ == "__main__":
    main()
