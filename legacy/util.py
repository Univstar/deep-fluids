import os
import math
import json
import logging
import numpy as np
from PIL import Image
from datetime import datetime
from glob import glob
import matplotlib.pyplot as plt

def prepare_dirs_and_logger(config):
    # print(__file__)
    os.chdir(os.path.dirname(__file__))

    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # data path
    config.data_path = os.path.join(config.data_dir, config.dataset)
    # config.data_path_disc = os.path.join(config.data_dir, config.dataset_disc)

    # model path
    if config.load_path:
        config.model_dir = config.load_path

    elif not hasattr(config, 'model_dir'):    
        model_name = "{}/{}_de_{}".format(
            config.dataset, get_time(), config.tag)

        config.model_dir = os.path.join(config.log_dir, model_name)
    
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False, flip=True):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            if flip: grid[h:h+h_width, w:w+w_width] = tensor[k,::-1]
            else: grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, scale_each=False, flip=True, single=False):
    if not single:
        ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                          normalize=normalize, scale_each=scale_each, flip=flip)
    else:
        h, w = tensor.shape[0], tensor.shape[1]
        ndarr = np.zeros([h,w,3], dtype=np.uint8)
        if flip: ndarr = tensor[::-1]
        else: ndarr = tensor
    
    im = Image.fromarray(ndarr)
    im.save(filename)
