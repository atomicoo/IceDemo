import sys, os
import glob

import torch
import yaml

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
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
    return model, optimizer, learning_rate, iteration


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
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
    config.data.min_text_len = getattr(config.data, "min_text_len", 1)
    config.data.max_text_len = getattr(config.data, "max_text_len", 190)

    config.data.n_languages = getattr(config.data, "n_languages", 0)
    config.data.n_speakers = getattr(config.data, "n_speakers", 0)
    config.data.n_styles = getattr(config.data, "n_styles", 0)

    config.data.skip_token_ids = getattr(config.data, "skip_token_ids", [])
    config.data.blank_both_ends = getattr(config.data, "blank_both_ends", False)

    config.model.synthesizer.z_utterance_dim = getattr(config.model.synthesizer, "z_utterance_dim", 0)

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

