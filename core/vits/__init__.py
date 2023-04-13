import os
import numpy as np
import torch
from . import utils
from . import models


def insert_blank_token(feat, ctx_group_idx):
    ret_size = len(feat) + ctx_group_idx[-1]
    head = 0
    result = np.zeros((ret_size, feat.shape[-1]), dtype=feat.dtype)
    result[:, 0] = 1  # NOTE: Blank phone is shared with padding phoneme.
    for i in range(len(feat)):
        if i > 0 and ctx_group_idx[i - 1] != ctx_group_idx[i]:
            head += 1
        result[head] = feat[i]
        head += 1
    return result


def load_model(checkpoint, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = os.path.join(checkpoint, "config.yaml")
    config = utils.load_config(config)

    if config.data.linguistic_feature_dim > 0:
        n_vocab = 0
        linguistic_feature_dim = config.data.linguistic_feature_dim
    else:
        n_vocab = config.data.n_vocab
        linguistic_feature_dim = 0
    model = models.Synthesizer(
        n_vocab,
        config.data.n_fft // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        n_languages=config.data.n_languages,
        n_speakers=config.data.n_speakers,
        n_styles=config.data.n_styles,
        linguistic_feature_dim=linguistic_feature_dim,
        **config.model.synthesizer,
    ).to(device=device)
    model.eval()

    checkpoint = os.path.join(checkpoint, "model")
    checkpoint = utils.latest_checkpoint_path(checkpoint)
    utils.load_checkpoint(checkpoint, model, None)
    print(f"Model has been loaded from {checkpoint}.")

    return model


@torch.no_grad()
def inference(model, text_input, z_u, lid=None, sid=None, tid=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_input = np.array(text_input)[..., None]
    text_input = insert_blank_token(text_input, np.arange(len(text_input)))
    text_input = torch.from_numpy(text_input).squeeze(-1)
    x_test = text_input.to(device).unsqueeze(0)
    x_test_lengths = torch.LongTensor([text_input.size(0)]).to(device)

    lid = torch.LongTensor([lid]).to(device) if lid is not None else None
    sid = torch.LongTensor([sid]).to(device) if sid is not None else None
    tid = torch.LongTensor([tid]).to(device) if tid is not None else None

    z_u = np.array(z_u, np.float32)
    z_u = torch.from_numpy(z_u).to(device).unsqueeze(0)

    outputs = model.infer(
        x_test,
        x_test_lengths,
        y=None,
        y_lengths=None,
        z_u=z_u,
        lid=lid,
        sid=sid,
        tid=tid,
        dur=None,
        noise_scale=0.667,
        noise_scale_w=0.8,
        length_scale=1,
        target_len=None,
        skip_mask=None,
        forced_skip=False,
    )

    audio = outputs[0][0, 0].data.cpu().float().numpy()

    return audio

