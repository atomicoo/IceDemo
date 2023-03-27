import sys, os
import importlib
import struct
import webrtcvad
from scipy.ndimage import morphology
import numpy as np
import torch

from . import utils

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging


def trim_long_silences(wave, sampling_rate, bit_depth=16, window_length=30, moving_average_width=8, max_silence_length=6):
    # Compute the voice detection window size
    samples_per_window = (window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wave = wave[:len(wave) - (len(wave) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    max_wav_value = float(2 ** (bit_depth - 1))
    pcm_wave = struct.pack("%dh" % len(wave), *(np.round(wave * max_wav_value)).astype(np.dtype(f'int{bit_depth}')))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wave), samples_per_window):
        window_end = window_start + samples_per_window
         # The WebRTC VAD only accepts 16-bit mono PCM audio, sampled at 8000, 16000, 32000 or 48000 Hz.
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2 : window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilate the voiced regions
    audio_mask = morphology.binary_dilation(audio_mask, np.ones(max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wave[audio_mask == True], audio_mask


class SpeakerNet(torch.nn.Module):
    def __init__(self, model, **kwargs):
        super(SpeakerNet, self).__init__()
        SpeakerNetModel = importlib.import_module("src.resnetse.models." + model).__getattribute__("MainModel")
        self.__S__ = SpeakerNetModel(**kwargs)

    def forward(self, data):
        data = data.reshape(-1, data.size()[-1])
        outp = self.__S__.forward(data)
        return outp


def load_model(checkpoint, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = os.path.join(checkpoint, "config.yaml")
    config = utils.load_config(config)

    model = SpeakerNet(**config).to(device)
    model.eval()

    checkpoint = utils.latest_checkpoint_path(checkpoint)
    utils.load_checkpoint(checkpoint, model)
    logger.info(f"Model has been loaded from {checkpoint}.")

    return model


def preprocess(audio, max_frames=400, num_eval=10):
    # Maximum audio length
    max_audio = max_frames * 160 + 240
    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1 
        audio       = np.pad(audio, (0, shortage), 'wrap')
        audiosize   = audio.shape[0]

    startframe = np.linspace(0, audiosize-max_audio, num=num_eval)
    
    feats = []
    if max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = np.stack(feats, axis=0).astype(np.float32)

    return feat


@torch.no_grad()
def inference(model, audio_data, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_data = preprocess(np.array(audio_data), max_frames=400)
    audio_data = torch.from_numpy(audio_data).to(device)

    outputs = model.forward(audio_data)
    embeddings = outputs.data.cpu().float().numpy()

    return embeddings

