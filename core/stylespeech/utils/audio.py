import numpy as np
import librosa
import struct
import webrtcvad
from scipy.ndimage.morphology import binary_dilation


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
    audio_mask = binary_dilation(audio_mask, np.ones(max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wave[audio_mask == True], audio_mask


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    C: compression factor
    """
    return np.log(np.clip(x, a_min=clip_val, a_max=float('inf')) * C)


def dynamic_range_decompression(x, C=1):
    """
    C: compression factor used to compress
    """
    return np.exp(x) / C


class MySTFT:
    def __init__(self):
        self.filter_length = 2048
        self.hop_length = 300
        self.win_length = 1200
        self.window = 'hann'
        self.power = 1.0
        self.n_mel_channels = 80
        self.sampling_rate = 24000
        self.mel_fmin = 0.0
        self.mel_fmax = 12000.0

        self.mel_basis = librosa.filters.mel(
            sr=self.sampling_rate, n_fft=self.filter_length, 
            n_mels=self.n_mel_channels, 
            fmin=self.mel_fmin, fmax=self.mel_fmax)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def wave_to_melspec(self, wave):
        stft_matrix = librosa.stft(
            wave, n_fft=self.filter_length, 
            hop_length=self.hop_length, win_length=self.win_length, 
            window=self.window)
        # magnitudes = np.abs(stft_matrix) ** self.power
        # phase = np.exp(1.0j * np.angle(stft_matrix))
        magnitudes, phase = librosa.magphase(stft_matrix, power=self.power)
        # melspec = np.matmul(self.mel_basis, magnitudes)
        melspec = np.einsum("...ft,mf->...mt", magnitudes, self.mel_basis, optimize=True)
        melspec = self.spectral_normalize(melspec)

        return melspec, magnitudes

    def melspec_to_wave(self, melspec, griffin_iters=60):
        melspec = self.spectral_de_normalize(melspec)
        inverse = librosa.util.nnls(self.mel_basis, melspec)
        magnitudes = np.power(inverse, 1.0 / self.power, out=inverse)
        wave = self.griffin_lim(magnitudes, n_iter=griffin_iters)

        return wave

    def griffin_lim(self, magnitudes, n_iter=32, momentum=0.99):
        # randomly initialize the phase
        angles = np.exp(2j * np.pi * np.random.rand(*magnitudes.shape)).astype(np.complex64)
        eps = librosa.util.tiny(angles)

        # And initialize the previous iterate to 0
        rebuilt = 0.0
        for _ in range(n_iter):
            # Store the previous iterate
            tprev = rebuilt
            # Invert with our current estimate of the phases
            inverse = librosa.istft(
                magnitudes * angles,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.hparams.window)
            # Rebuild the spectrogram
            rebuilt = librosa.stft(
                inverse,
                n_fft=self.filter_length,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window)
            # Update our phase estimates
            angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
            angles[:] /= np.abs(angles) + eps

        return librosa.istft(
            magnitudes * angles,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window)