from torch import nn
from typing import Tuple
import torch

from model.nn.modules import (EmbeddingWithPosition, TransformerEncoderLayer)

class TransformerVAE(nn.Module):
    def __init__(self,
                 max_seq_len: int,
                 vocab_size: int,
                 embedding_weights=None,
                 num_layers=10,
                 emb_size=768,
                 latent_size=150,
                 dim_m=512,
                 dim_i=2048,
                 n_heads=12,
                 initial_token_idx=101,
                 dropout=0.1):

        super(TransformerVAE, self).__init__()

        self.initial_token_idx = initial_token_idx

        if embedding_weights is not None:
            assert isinstance(embedding_weights, torch.Tensor), "embedding must be a torch.Tensor"
            vocab_size, emb_size = embedding_weights.shape
        self.vocab_size = vocab_size

        message = 'Model `dim_m` must be divisible by `n_heads` without a remainder.'
        assert dim_m % n_heads == 0, message
        dim_proj = dim_m // n_heads

        encoder_decoder_args = {
            'dim_m': dim_m,
            'dim_q_k': dim_proj,
            'dim_v': dim_proj,
            'n_heads': n_heads,
            'dim_i': dim_i,
            'dropout': dropout
        }

        # Transformer
        self.embedding = EmbeddingWithPosition(max_seq_len, dim_m, vocab_size, emb_size, embedding_weights)
        # Encoder
        self.repr_token = nn.Embedding(1, dim_m)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(**encoder_decoder_args) for _ in range(num_layers)
        ])
        # Decoder
        # 这里具体换掉
        # self.decoder_layers = nn.ModuleList([
        #     TransformerDecoderLayer(**encoder_decoder_args) for _ in range(num_layers)
        # ])
        # self.out = nn.Sequential(
        #     nn.Linear(dim_m, vocab_size)
        # )

        # VAE
        self.to_mu = nn.Linear(dim_m, latent_size)
        self.to_logvar = nn.Linear(dim_m, latent_size)
        # 这一层是为了做维度兼容
        self.decode_latent = nn.Linear(latent_size, dim_m)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = input.shape[0]
        input_embedded = self.embedding(input)


        encoder_state = input_embedded

        for encoder_layer in self.encoder_layers:
            encoder_state = encoder_layer(encoder_state)

        # 这里用mlp或者last hidden state as sequence context vector
        # if not self.pool_context:
        #     # Use last hidden state as sequence context vector:
        #     seq_repr = encoder_state[:, -1, :].view(batch_size, -1)
        # else:
        #     seq_repr = encoder_state.mean(dim=1).view(batch_size, -1)

        # Reparameterize
        mu = self.to_mu(seq_repr)
        logvar = self.to_logvar(seq_repr)
        z = self.reparameterize(mu, logvar)

        encoder_state = self.decode_latent(z).view(batch_size, 1, -1)

        # Decode 这里对应换掉
        # mask = self.autoregressive_mask(input)
        # decoder_state = input_embedded
        # for decoder_layer in self.decoder_layers:
        #     decoder_state = decoder_layer(decoder_state, encoder_state, mask)
        #
        # output = self.out(decoder_state)[:, :-1, :]
        return output.contiguous(), mu, logvar