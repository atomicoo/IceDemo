import sys, os
import glob

import torch
import yaml

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging


def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    loaded_state = torch.load(checkpoint_path, map_location="cpu")
    self_state = model.state_dict()
    if len(loaded_state.keys()) == 1 and "model" in loaded_state:
        loaded_state = loaded_state["model"]
        newdict = {}
        delete_list = []
        for name, param in loaded_state.items():
            new_name = "__S__."+name
            newdict[new_name] = param
            delete_list.append(name)
        loaded_state.update(newdict)
        for name in delete_list:
            del loaded_state[name]
    for name, param in loaded_state.items():
        origname = name
        if name not in self_state:
            name = name.replace("module.", "")
            if name not in self_state:
                logger.info("{} is not in the model.".format(origname))
                continue
        if self_state[name].size() != loaded_state[origname].size():
            logger.info("Wrong parameter length: {}, model: {}, loaded: {}".format(
                origname, self_state[name].size(), loaded_state[origname].size()))
            continue
        self_state[name].copy_(param)
    logger.info("Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, 0))
    return model


def latest_checkpoint_path(dir_path, regex="*.model"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    logger.info("Latest checkpoint path: {}".format(x))
    return x


def load_config(filepath):
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)
    config = Config(**config)
    config = keep_compatibility(config)
    return config


def keep_compatibility(config):
    config.eval_frames = getattr(config, "eval_frames", 400)

    config.n_mels = getattr(config, "n_mels", 60)
    config.log_input = getattr(config, "log_input", True)
    config.model = getattr(config, "model", "ResNetSE34V2")
    config.encoder_type = getattr(config, "encoder_type", "ASP")
    config.nOut = getattr(config, "nOut", 512)

    return config


class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = Config(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
