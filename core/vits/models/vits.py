from typing import Dict, Optional

import torch
from torch import nn

from .functions import generate_path, sequence_mask
from .modules.duration_predictor import DurationPredictor, StochasticDurationPredictor
from .modules.encoders import ConvTextEncoder, PosteriorEncoder, TextEncoder, UtteranceEncoder
from .modules.flow import ResidualCouplingBlock
from .modules.hifigan import Generator


class Synthesizer(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        text_encoder_params: Dict[str, any] = {},
        generator_params: Dict[str, any] = {},
        flow_params: Dict[str, any] = {},
        n_languages: int = 0,
        n_speakers: int = 0,
        n_styles: int = 0,
        linguistic_feature_dim: int = 0,
        z_utterance_dim: int = 0,
        utterance_encoder_params: Dict[str, any] = {},
        gmvae_params: Optional[Dict[str, any]] = None,
        cin_channels: int = 0,
        gin_channels: int = 0,
        skip_factor: float = 1.0,
        use_sdp: bool = True,
        use_classifier: bool = False,
        text_encoder_type: str = "MHSA",
    ):

        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.segment_size = segment_size
        self.n_languages = n_languages
        self.n_speakers = n_speakers
        self.n_styles = n_styles
        self.linguistic_feature_dim = linguistic_feature_dim
        self.z_utterance_dim = z_utterance_dim
        self.cin_channels = cin_channels
        self.gin_channels = gin_channels
        self.skip_factor = skip_factor

        if n_languages + n_speakers + n_styles + z_utterance_dim == 0:
            assert gin_channels == 0, f"gin_channels should be 1 but got {gin_channels}"
        else:
            assert gin_channels > 0, f"gin_channels should be greater than 1 but got {gin_channels}"

        self.use_sdp = use_sdp
        self.use_classifier = use_classifier

        if text_encoder_type == "MHSA":
            self.enc_p = TextEncoder(
                n_vocab,
                linguistic_feature_dim,
                inter_channels,
                hidden_channels,
                **text_encoder_params,
            )
        elif text_encoder_type == "conv":
            self.enc_p = ConvTextEncoder(
                n_vocab,
                linguistic_feature_dim,
                inter_channels,
                hidden_channels,
                **text_encoder_params,
            )
        else:
            raise ValueError(f"Unknown text_encoder_type {text_encoder_type}")

        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            gin_channels=gin_channels if n_languages + n_speakers + n_styles > 0 else 0,
        )

        self.dec = Generator(
            inter_channels,
            gin_channels=gin_channels,
            **generator_params,
        )

        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            cin_channels=cin_channels,
            gin_channels=gin_channels,
            **flow_params,
        )

        if z_utterance_dim > 0:
            self.enc_u = UtteranceEncoder(
                inter_channels,
                z_utterance_dim,
                gin_channels=gin_channels if n_languages + n_speakers + n_styles > 0 else 0,
                **utterance_encoder_params,
            )
            self.proj_u = nn.Linear(z_utterance_dim, gin_channels)

        if use_sdp:
            self.dp = StochasticDurationPredictor(hidden_channels, gin_channels=gin_channels)
        else:
            self.dp = DurationPredictor(hidden_channels, gin_channels=gin_channels)

        if n_languages > 0:
            self.emb_lang = nn.Embedding(n_languages, gin_channels)
        if n_speakers > 0:
            self.emb_spk = nn.Embedding(n_speakers, gin_channels)
        if n_styles > 0:
            self.emb_sty = nn.Embedding(n_styles, gin_channels)

    def _embedding(self, lid=None, sid=None, tid=None):
        g = None
        if self.n_languages > 0:
            g = self.emb_lang(lid).unsqueeze(-1)  # [b, h, 1]
        if self.n_speakers > 0:
            if g is not None:
                g += self.emb_spk(sid).unsqueeze(-1)
            else:
                g = self.emb_spk(sid).unsqueeze(-1)
        if self.n_styles > 0:
            if g is not None:
                g += self.emb_sty(tid).unsqueeze(-1)
            else:
                g = self.emb_sty(tid).unsqueeze(-1)
        return g

    def infer(
        self,
        x,
        x_lengths,
        y=None,
        y_lengths=None,
        z_u=None,
        lid=None,
        sid=None,
        tid=None,
        c=None,
        dur=None,
        noise_scale=1,
        length_scale=1,
        noise_scale_w=1.0,
        target_len=None,
        max_len=None,
        skip_mask=None,
        forced_skip=False,
    ):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        g = self._embedding(lid, sid, tid)

        if self.z_utterance_dim > 0:
            if y is not None and y_lengths is not None:
                z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
                z_u, m_u, logs_u = self.enc_u(z, y_lengths, g=g)
            elif z_u is None:
                z_u = torch.randn(x.size(0), self.z_utterance_dim).to(x.device)
                m_u, logs_u = None, None
            else:
                m_u, logs_u = None, None
            if g is not None:
                g += self.proj_u(z_u).unsqueeze(-1)
            else:
                g = self.proj_u(z_u).unsqueeze(-1)
        else:
            z_u, m_u, logs_u = None, None, None

        if dur is None:
            if self.use_sdp:
                logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
            else:
                logw = self.dp(x, x_mask, g=g)
            w = torch.exp(logw) * x_mask * length_scale
            if target_len is not None:
                w *= target_len / torch.sum(w, [1, 2])
            if skip_mask is not None:
                if forced_skip:
                    w_ceil = torch.where(skip_mask, torch.zeros_like(w), torch.ceil(w))
                else:
                    w_ceil = torch.where(skip_mask, torch.floor(w), torch.ceil(w))
            else:
                w_ceil = torch.ceil(w)
        else:
            w_ceil = dur.unsqueeze(1)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)

        # expand prior [b, t', t], [b, t, d] -> [b, d, t']
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, c=c, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p, z_u, m_u, logs_u)

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.emb_spk(sid_src).unsqueeze(-1)
        g_tgt = self.emb_spk(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)

