from torch import nn
import torch


from model.nn.modules import (EmbeddingWithPosition, TransformerEncoderLayer)


import torch.nn.functional as F


pad_idx = 0
unk_idx = 100
cls_idx = 101
sep_idx = 102


class TransformerBOW(nn.Module):
    def __init__(self,
                 n_ctx: int,
                 vocab_size: int,
                 embedding_weights=None,
                 num_layers=10,
                 emb_size=768,
                 latent_size=768,
                 dim_m=512,
                 dim_i=2048,
                 n_heads=8,
                 initial_token_idx=101,
                 dropout=0.1,
                 repr_form="last"):

        super(TransformerBOW, self).__init__()

        self.initial_token_idx = initial_token_idx

        if embedding_weights is not None:
            assert isinstance(embedding_weights, torch.Tensor), "embedding must be a torch.Tensor"
            vocab_size, emb_size = embedding_weights.shape
        self.vocab_size = vocab_size
        self.repr_form = repr_form

        message = 'Model `dim_m` must be divisible by `n_heads` without a remainder.'
        assert dim_m % n_heads == 0, message
        dim_proj = dim_m // n_heads

        encoder_args = {
            'dim_m': dim_m,
            'dim_q_k': dim_proj,
            'dim_v': dim_proj,
            'n_heads': n_heads,
            'dim_i': dim_i,
            'dropout': dropout
        }

        # Transformer
        self.embedding = EmbeddingWithPosition(n_ctx, dim_m, vocab_size, emb_size, embedding_weights)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(**encoder_args) for _ in range(num_layers)
        ])

        # VAE
        # first-last
        if self.repr_form == "fl":
            self.to_mu = nn.Linear(2 * dim_m, latent_size)
            self.to_logvar = nn.Linear(2 * dim_m, latent_size)
        # first-mean-last
        elif self.repr_form == "fml":
            self.to_mu = nn.Linear(3 * dim_m, latent_size)
            self.to_logvar = nn.Linear(3 * dim_m, latent_size)
        else:
            self.to_mu = nn.Linear(dim_m, latent_size)
            self.to_logvar = nn.Linear(dim_m, latent_size)

        self.bow_predictor = nn.Linear(latent_size, vocab_size)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, input: torch.Tensor):
        batch_size = input.shape[0]
        input_embedded = self.embedding(input)

        encoder_state = input_embedded

        for encoder_layer in self.encoder_layers:
            encoder_state = encoder_layer(encoder_state)

        # Use last hidden state as sequence context vector:
        # 用最后一个状态可能太弱了，试下换成均值
        # 按照原来的用均值？用最后一个？
        # 后续用头来做z？改维度，用头concate尾？头-尾-均值 concate？
        if self.repr_form == "last":
            seq_repr = encoder_state[:, -1, :].view(batch_size, -1)
        elif self.repr_form == "mean":
            seq_repr = encoder_state.mean(dim=1).view(batch_size, -1)
        elif self.repr_form == "first":
            seq_repr = encoder_state[:, 0, :].view(batch_size, -1)
        elif self.repr_form == "fl":
            seq_repr = torch.cat((encoder_state[:, 0, :], encoder_state[:, -1, :]), -1).view(batch_size, -1)
        elif self.repr_form == "fml":
            seq_repr = torch.cat((encoder_state[:, 0, :], encoder_state.mean(dim=1), encoder_state[:, -1, :]), -1).view(batch_size, -1)

        # Reparameterize
        mu = self.to_mu(seq_repr)
        logvar = self.to_logvar(seq_repr)
        z = self.reparameterize(mu, logvar)

        # bow
        bow_logits = self.bow_predictor(z)
        bow_probs = F.softmax(bow_logits, dim=-1)

        return mu, logvar, bow_probs