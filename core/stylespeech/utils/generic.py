import sys, os
import glob

import torch
import json

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging


def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["step"]
    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except KeyError:
            logger.info("{} is not in the checkpoint".format(k))
            new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    logger.info("Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration))
    return model, iteration


def latest_checkpoint_path(dir_path, regex="*.pth.tar"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    logger.info("Latest checkpoint path: {}".format(x))
    return x


def load_config(filepath):
    with open(filepath) as f:
        json_config = json.loads(f.read())
    config = AttrDict(json_config)
    return config


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

