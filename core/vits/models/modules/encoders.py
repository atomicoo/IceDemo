import math

import torch
from torch import nn
from torch.nn.utils import rnn

from ..functions import sequence_mask
from . import attentions, layers


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        linguistic_feature_dim: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: int,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.linguistic_feature_dim = linguistic_feature_dim
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        if linguistic_feature_dim == 0:
            self.emb = nn.Embedding(n_vocab, hidden_channels)
        else:
            self.emb = nn.Linear(linguistic_feature_dim, hidden_channels)

        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ConvTextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        linguistic_feature_dim: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: int,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.linguistic_feature_dim = linguistic_feature_dim
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        if linguistic_feature_dim == 0:
            self.emb = nn.Embedding(n_vocab, hidden_channels)
        else:
            self.emb = nn.Linear(linguistic_feature_dim, hidden_channels)

        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        sizes = [hidden_channels] * (n_layers + 1)
        convs = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            convs.append(
                nn.Sequential(
                    layers.GatedTanhConv1d(in_size, out_size, kernel_size),
                    nn.Dropout(p_dropout),
                )
            )

        self.encoder = nn.Sequential(*convs)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 16,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = layers.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class UtteranceEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 512,
        kernel_size: int = 3,
        lstm_layers: int = 2,
        lstm_dim: int = 256,
        gin_channels: int = 0,
        detach_input: bool = False,
    ):
        super().__init__()
        in_channels += gin_channels
        self.out_channels = out_channels
        self.detach_input = detach_input

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=(kernel_size - 1) // 2),
        )
        self.lstm = nn.LSTM(hidden_channels, lstm_dim, lstm_layers, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(lstm_dim * 2, out_channels * 2)

    def forward(self, x, x_lengths, g=None):
        if g is not None:
            x = torch.cat([x, g.expand(-1, -1, x.size(2))], 1)

        if self.detach_input:
            x = x.detach()

        x = self.conv(x)  # (B, channels, T)
        x = x.transpose(1, 2)  # (B, T, channels)
        x = rnn.pack_padded_sequence(x, x_lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = rnn.pad_packed_sequence(x, batch_first=True)  # (B, T, 2 * lstm_dim)
        x = torch.sum(x, dim=-2) / x_lengths.unsqueeze(1)  # (B, 2 * lstm_dim)
        stats = self.proj(x)
        m, logs = torch.split(stats, self.out_channels, dim=1)  # (B, output_dim)
        z = m + torch.randn_like(m) * torch.exp(logs)
        return z, m, logs
