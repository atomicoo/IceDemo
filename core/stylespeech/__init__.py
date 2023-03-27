import os
import struct
import webrtcvad
from scipy.ndimage import morphology
import numpy as np
import torch
from . import utils
from .models import Synthesizer
from core.hifigan import HiFiGANGenerator
from . import audio as Audio


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


def load_model(checkpoint, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = os.path.join(checkpoint, "config.json")
    config = utils.load_config(config)

    model = Synthesizer(config).to(device=device)
    model.eval()

    checkpoint = os.path.join(checkpoint, "model")
    checkpoint = utils.latest_checkpoint_path(checkpoint)
    utils.load_checkpoint(checkpoint, model)
    print(f"Model has been loaded from {checkpoint}.")

    model.stft = Audio.stft.MySTFT()

    vocoder = HiFiGANGenerator()
    checkpoint = os.path.join(os.getcwd(), 'core', 'hifigan', 'pretrain.pth')
    checkpoint = torch.load(checkpoint, map_location="cpu")
    vocoder.load_state_dict(checkpoint['model'])
    vocoder.register_stats(checkpoint['stats'])
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to(device)
    model.vocoder = vocoder

    return model


@torch.no_grad()
def inference(model, text_input, ref_audio, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_input = np.array(text_input)[None, ...]
    src = torch.from_numpy(text_input).to(device)
    src_len = torch.LongTensor([text_input.shape[0]]).to(device)

    # style_vector = torch.randn(1, 128).float().to(device)
    # wave, _ = trim_long_silences(ref_audio, sampling_rate=24_000)
    wave = torch.FloatTensor(ref_audio.astype(np.float32))
    ref_mel, _ = model.stft.mel_spectrogram(wave.unsqueeze(0))
    ref_mel = ref_mel.transpose(1,2).to(device)
    style_vector = model.get_style_vector(ref_mel)

    melspec = model.inference(style_vector, src, src_len)[0][0]
    waveform = model.vocoder.inference(melspec, normalize_before=True).view(-1)

    audio = waveform.data.cpu().float().numpy()
    melspec = melspec.data.cpu().float().numpy()

    return audio, melspec

