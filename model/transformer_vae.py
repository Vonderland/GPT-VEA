from torch import nn
from typing import Tuple
import torch
import transformers


from model.nn.modules import (EmbeddingWithPosition, TransformerEncoderLayer)
from model.nn.gpt2 import GPT2LMHeadModel
from transformers.modeling_gpt2 import GPT2Config



class TransformerVAE(nn.Module):
    def __init__(self,
                 n_ctx: int,
                 vocab_size: int,
                 embedding_weights=None,
                 pretrained_decoder=None,
                 decoder_config=None,
                 num_layers=10,
                 emb_size=768,
                 latent_size=768,
                 dim_m=768,
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
        # Decoder
        # 这里具体换掉
        if pretrained_decoder:  # 如果指定了预训练的GPT2模型
            self.decoder = GPT2LMHeadModel.from_pretrained(pretrained_decoder)
        else:  # 若没有指定预训练模型，则初始化模型
            decoder_config = transformers.modeling_gpt2.GPT2Config.from_json_file(decoder_config)
            self.decoder = GPT2LMHeadModel(config=decoder_config)
        # 根据tokenizer的vocabulary调整GPT2模型的vocal的大小
        self.decoder.resize_token_embeddings(vocab_size)
        print('model config:\n{}'.format(self.decoder.config.to_json_string()))

        # VAE
        self.to_mu = nn.Linear(dim_m, latent_size)
        self.to_logvar = nn.Linear(dim_m, latent_size)

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

        # Use last hidden state as sequence context vector:
        seq_repr = encoder_state[:, -1, :].view(batch_size, -1)


        # Reparameterize
        mu = self.to_mu(seq_repr)
        logvar = self.to_logvar(seq_repr)
        z = self.reparameterize(mu, logvar)
        z = z.view(batch_size, 1, -1)

        # 这里的z是每个token都会加的！！！需要变换和拼接而且不应该用embedding而是应该用linear！！而且这个中文参考代码好像有问题，这里需要传mask进去的
        # 中文的没传mask进去是因为他自己另外实现了calculate_loss_and_accuracy里用label跳过了mask
        outputs = self.decoder.forward(input_ids=input, latent_z=z)

        return outputs, mu, logvar