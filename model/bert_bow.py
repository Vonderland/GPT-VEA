from torch import nn
import torch

from transformers import BertModel, configuration_utils

import torch.nn.functional as F


class BertBOW(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 latent_size=768,
                 dim_m=512,
                 repr_form="last"):

        super(BertBOW, self).__init__()


        # Transformer
        model_config = configuration_utils.PretrainedConfig.from_json_file("config/bert_config.json")
        self.encoder = BertModel(config=model_config)
        self.repr_form = repr_form

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
        encoder_state = self.encoder.forward(input_ids=input)[0]

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
            seq_repr = torch.cat((encoder_state[:, 0, :], encoder_state.mean(dim=1), encoder_state[:, -1, :]), -1).view(
                batch_size, -1)

        # Reparameterize
        mu = self.to_mu(seq_repr)
        logvar = self.to_logvar(seq_repr)
        z = self.reparameterize(mu, logvar)

        # bow
        bow_logits = self.bow_predictor(z)
        bow_probs = F.softmax(bow_logits, dim=-1)

        return mu, logvar, bow_probs