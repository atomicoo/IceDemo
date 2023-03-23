import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAveragePooling(nn.Module):
    def __init__(self, **kwargs):
        """TAP
        Paper: Multi-Task Learning with High-Order Statistics for X-vector based Text-Independent Speaker Verification
        Link: https://arxiv.org/pdf/1903.12058.pdf
        """
        super(TemporalAveragePooling, self).__init__()

    def forward(self, x):
        """Computes Temporal Average Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels)
        """
        x = torch.mean(x, dim=2)
        return x


class GlobalAveragePooling(nn.Module):
    def __init__(self, **kwargs):
        super(GlobalAveragePooling, self).__init__()
        self.global_average_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """Computes Global Average Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, dim, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, dim)
        """
        if x.ndim == 3:
            x = x.unsqueeze(dim=2)
        x = self.global_average_pooling(x)
        x = x.squeeze(2).squeeze(2)
        return x


class TemporalStatisticsPooling(nn.Module):
    def __init__(self, **kwargs):
        """TSP
        Paper: X-vectors: Robust DNN Embeddings for Speaker Recognition
        Link: http://www.danielpovey.com/files/2018_icassp_xvectors.pdf
        """
        super(TemporalStatisticsPooling, self).__init__()

    def forward(self, x):
        """Computes Temporal Statistics Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, channels*2)
        """
        mean = torch.mean(x, dim=2)
        var = torch.var(x, dim=2)
        x = torch.cat((mean, var), dim=1)
        return x


class SelfAttentivePooling(nn.Module):
    def __init__(self, dim, **kwargs):
        """SAP
        Paper: Self-Attentive Speaker Embeddings for Text-Independent Speaker Verification
        Link: https://danielpovey.com/files/2018_interspeech_xvector_attention.pdf
        Args:
            dim (pair): the size of attention weights
        """
        super(SelfAttentivePooling, self).__init__()
        self.sap_linear = nn.Linear(dim, dim)
        self.att_linear = nn.Linear(dim, 1)

    def forward(self, x):
        """Computes Self-Attentive Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, dim, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, dim)
        """
        x = x.permute(0, 2, 1)
        h = torch.tanh(self.sap_linear(x))
        w = self.att_linear(h).squeeze(dim=2)
        w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
        x = torch.sum(x * w, dim=1)
        return x


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, dim, **kwargs):
        """ASP
        Paper: Attentive Statistics Pooling for Deep Speaker Embedding
        Link: https://arxiv.org/pdf/1803.10963.pdf
        Args:
            dim (pair): the size of attention weights
        """
        super(AttentiveStatisticsPooling, self).__init__()
        self.sap_linear = nn.Linear(dim, dim)
        self.att_linear = nn.Linear(dim, 1)

    def forward(self, x):
        """Computes Attentive Statistics Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, dim, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, dim*2)
        """
        x = x.permute(0, 2, 1)
        h = torch.tanh(self.sap_linear(x))
        w = self.att_linear(h).squeeze(dim=2)
        w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
        mu = torch.sum(x * w, dim=1)
        rh = torch.sqrt((torch.sum((x**2) * w, dim=1) - mu**2).clamp(min=1e-5))
        x = torch.cat((mu, rh), dim=1)
        return x


class MultiHeadSelfAttentivePooling(nn.Module):
    def __init__(self, dim, num_heads=4, **kwargs):
        """MQSAP
        Paper: Self-Attentive Speaker Embeddings for Text-Independent Speaker Verification
        Link: https://danielpovey.com/files/2018_interspeech_xvector_attention.pdf
        """
        super(MultiHeadSelfAttentivePooling, self).__init__()
        self.num_heads = num_heads

        self.sap_linear = nn.Linear(dim, dim, bias=True)
        self.att_linear = nn.Linear(dim, num_heads, bias=False)

        # self.final_linear = nn.Linear(dim * num_heads, dim)

    def forward(self, x):
        """Computes MultiHead Self-Attentive Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, dim, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, dim)
        """
        x = x.permute(0, 2, 1)
        h = F.sigmoid(self.sap_linear(x))
        w = F.softmax(self.att_linear(h), dim=1)

        x = torch.einsum('btc,bth->bch', x, w)
        # return self.final_linear(x.view(x.size(0), -1))
        return torch.mean(x, dim=2)


class MultiHeadSelfAttentiveStatisticsPooling(nn.Module):
    def __init__(self, dim, num_heads=4, **kwargs):
        """MQASP"""
        super(MultiHeadSelfAttentiveStatisticsPooling, self).__init__()
        self.num_heads = num_heads

        self.sap_linear = nn.Linear(dim, dim, bias=True)
        self.att_linear = nn.Linear(dim, num_heads, bias=False)

    def forward(self, x):
        """Computes MultiHead Self-Attentive Statistics Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, dim, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, dim*2)
        """
        x = x.permute(0, 2, 1)
        h = F.sigmoid(self.sap_linear(x))
        w = F.softmax(self.att_linear(h), dim=1)

        mu = torch.mean(torch.einsum('btc,bth->bch', x, w), dim=2)
        rh = torch.sqrt(
            (torch.mean(torch.einsum('btc,bth->bch', x**2, w), dim=2) - mu**2)
            .clamp(min=1e-5))
        x = torch.cat((mu, rh), dim=1)
        return x


class SelfMultiHeadAttentionPooling(nn.Module):
    def __init__(self, dim, num_heads=4, **kwargs):
        """MHSAP
        Paper: Self Multi-Head Attention for Speaker Recognition
        Link: https://arxiv.org/pdf/1906.09890.pdf
        """
        super(SelfMultiHeadAttentionPooling, self).__init__()
        self.num_heads = num_heads

        self.attentions = nn.Sequential(*[
            SelfAttentivePooling(dim//num_heads) for _ in range(num_heads)])

    def forward(self, x):
        """Computes Self MultiHead Attention Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, dim, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, dim)
        """
        p = torch.split(x, x.size(1)//self.num_heads, dim=1)
        xx = [self.attentions[i](p[i]) for i in range(self.num_heads)]
        x = torch.cat(xx, dim=1)
        return x


class SelfMultiHeadAttentionStatisticsPooling(nn.Module):
    def __init__(self, dim, num_heads=4, **kwargs):
        """MHASP"""
        super(SelfMultiHeadAttentionStatisticsPooling, self).__init__()
        self.num_heads = num_heads

        self.attentions = nn.Sequential(*[
            AttentiveStatisticsPooling(dim//num_heads) for _ in range(num_heads)])

    def forward(self, x):
        """Computes Self MultiHead Attention Statistics Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, dim, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, dim*2)
        """
        p = torch.split(x, x.size(1)//self.num_heads, dim=1)
        xx = [self.attentions[i](p[i]).view(x.size(0), 2, -1) for i in range(self.num_heads)]
        x = torch.cat(xx, dim=2).view(x.size(0), -1)
        return x


class MultiQueryMultiHeadAttentionPooling(nn.Module):
    def __init__(self, dim, num_queries=4, num_heads=16, **kwargs):
        """MQMHSAP
        Paper: The SpeakIn System for VoxCeleb Speaker Recognition Challange 2021
        Link: https://arxiv.org/pdf/2109.01989.pdf
        """
        super(MultiQueryMultiHeadAttentionPooling, self).__init__()
        self.num_queries = num_queries
        self.num_heads = num_heads

        self.attentions = nn.Sequential(*[
            MultiHeadSelfAttentivePooling(dim//num_heads, num_heads=num_queries) for _ in range(num_heads)])

    def forward(self, x):
        """Computes MultiQuery MultiHead Attentive Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, dim, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, dim)
        """
        p = torch.split(x, x.size(1)//self.num_heads, dim=1)
        xx = [self.attentions[i](p[i]) for i in range(self.num_heads)]
        x = torch.cat(xx, dim=1)

        return x


class MultiQueryMultiHeadAttentionStatisticsPooling(nn.Module):
    def __init__(self, dim, num_queries=4, num_heads=16, **kwargs):
        """MQMHASP"""
        super(MultiQueryMultiHeadAttentionStatisticsPooling, self).__init__()
        self.num_queries = num_queries
        self.num_heads = num_heads

        self.attentions = nn.Sequential(*[
            MultiHeadSelfAttentiveStatisticsPooling(dim//num_heads, num_heads=num_queries) for _ in range(num_heads)])

    def forward(self, x):
        """Computes MultiQuery MultiHead Attentive Statistics Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, dim, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, dim*2)
        """
        p = torch.split(x, x.size(1)//self.num_heads, dim=1)
        xx = [self.attentions[i](p[i]).view(x.size(0), 2, -1) for i in range(self.num_heads)]
        x = torch.cat(xx, dim=2).view(x.size(0), -1)
        return x


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, dim, num_heads=4, epsilon=1e-7, **kwargs):
        """MHAP
        Paper: PSLA: Improving Audio Tagging with Pretraining, Sampling, Labeling, and Aggregation
        Link: https://arxiv.org/pdf/2102.01243.pdf
        """
        super(MultiHeadAttentionPooling, self).__init__()
        self.num_heads = num_heads
        self.epsilon = epsilon

        self.clf_linear = nn.Linear(dim, dim * num_heads, bias=True)
        self.att_linear = nn.Linear(dim, dim * num_heads, bias=True)

        self.mha_weight = nn.Parameter(torch.FloatTensor([[1.0/num_heads]] * num_heads))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        c = F.sigmoid(self.clf_linear(x))
        a = F.sigmoid(self.att_linear(x))

        a = torch.clamp(a, self.epsilon, 1.0-self.epsilon)
        w = a / torch.sum(a, dim=1, keepdim=True)
        x = torch.sum(c * w, dim=1)

        x = x.view(x.size(0), self.num_heads, -1) * self.mha_weight
        x = torch.sum(x, dim=1).squeeze(dim=1)
        return x


if __name__ == "__main__":
    data = torch.randn(8, 80, 99)
    pooling = MultiQueryMultiHeadAttentionStatisticsPooling(80)
    out = pooling(data)
    print(data.shape)
    print(out.shape)
