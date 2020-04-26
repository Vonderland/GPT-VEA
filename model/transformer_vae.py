from torch import nn
from typing import Tuple
import torch
import transformers


from model.nn.modules import (EmbeddingWithPosition, TransformerEncoderLayer)
from model.nn.gpt2 import GPT2LMHeadModel
from transformers.modeling_gpt2 import GPT2Config
from model.nn.gpt2 import configuration_utils


import torch.nn.functional as F


pad_idx = 0
unk_idx = 100
cls_idx = 101
sep_idx = 102


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class TransformerVAE(nn.Module):
    def __init__(self,
                 n_ctx: int,
                 vocab_size: int,
                 embedding_weights=None,
                 pretrained_decoder=None,
                 decoder_config=None,
                 use_avg=True,
                 with_bow=True,
                 num_layers=10,
                 emb_size=768,
                 latent_size=768,
                 dim_m=768,
                 dim_i=2048,
                 n_heads=12,
                 initial_token_idx=101,
                 dropout=0.1,
                 word_dropout=0):

        super(TransformerVAE, self).__init__()

        self.initial_token_idx = initial_token_idx

        if embedding_weights is not None:
            assert isinstance(embedding_weights, torch.Tensor), "embedding must be a torch.Tensor"
            vocab_size, emb_size = embedding_weights.shape
        self.vocab_size = vocab_size
        self.word_dropout_rate = word_dropout
        self.use_avg = use_avg
        self.with_bow = with_bow

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
            print("use config, random initialization")
            model_config = configuration_utils.PretrainedConfig.from_json_file(decoder_config)
            self.decoder = GPT2LMHeadModel(config=model_config)
        # 根据tokenizer的vocabulary调整GPT2模型的vocal的大小
        self.decoder.resize_token_embeddings(vocab_size)
        print('model config:\n{}'.format(self.decoder.config.to_json_string()))

        # VAE
        self.to_mu = nn.Linear(dim_m, latent_size)
        self.to_logvar = nn.Linear(dim_m, latent_size)

        if self.with_bow:
            self.bow_predictor = nn.Linear(latent_size, vocab_size)


    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def save_decoder(self, decoder_path):
        self.decoder.save_pretrained(decoder_path)

    def inference(self, input: torch.Tensor, device, max_len, topk, topp):
        batch_size = input.shape[0]
        input_embedded = self.embedding(input)

        encoder_state = input_embedded

        for encoder_layer in self.encoder_layers:
            encoder_state = encoder_layer(encoder_state)

        # Use last hidden state as sequence context vector:
        # 用最后一个状态可能太弱了，试下换成均值
        if self.use_avg:
            seq_repr = encoder_state[:, -1, :].view(batch_size, -1)
        else:
            seq_repr = encoder_state.mean(dim=1).view(batch_size, -1)

        # Reparameterize
        mu = self.to_mu(seq_repr)
        logvar = self.to_logvar(seq_repr)
        z = self.reparameterize(mu, logvar)
        z = z.view(batch_size, 1, -1)

        generated = []
        input_ids = [cls_idx]
        curr_input_tensor = torch.tensor(input_ids).long().to(device)

        # 最多生成max_len个token
        for _ in range(max_len):
            outputs = self.decoder(input_ids=curr_input_tensor, latent_z=z)
            next_token_logits = outputs[0][-1, :]
            # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
            next_token_logits[unk_idx] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)
            # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            # if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
            #     break
            generated.append(next_token.item())
            curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=0)
            # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
            # print("his_text:{}".format(his_text))

        return generated

    def forward(self, input: torch.Tensor):
        batch_size = input.shape[0]
        input_embedded = self.embedding(input)

        encoder_state = input_embedded

        for encoder_layer in self.encoder_layers:
            encoder_state = encoder_layer(encoder_state)

        # Use last hidden state as sequence context vector:
        # 用最后一个状态可能太弱了，试下换成均值
        if self.use_avg:
            seq_repr = encoder_state[:, -1, :].view(batch_size, -1)
        else:
            seq_repr = encoder_state.mean(dim=1).view(batch_size, -1)


        # Reparameterize
        mu = self.to_mu(seq_repr)
        logvar = self.to_logvar(seq_repr)
        z = self.reparameterize(mu, logvar)

        bow_probs = None
        # bow
        if self.with_bow:
            bow_logits = self.bow_predictor(z)
            bow_probs = F.softmax(bow_logits, dim=-1)

        z = z.view(batch_size, 1, -1)


        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input.size())
            if torch.cuda.is_available():
                prob = prob.cuda()
            prob[(input.data - cls_idx) * (input.data - pad_idx) * (input.data - sep_idx) == 0] = 1
            decoder_input_sequence = input.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = unk_idx
            outputs = self.decoder.forward(input_ids=decoder_input_sequence, latent_z=z)
        else:
            outputs = self.decoder.forward(input_ids=input, latent_z=z)


        return outputs, mu, logvar, bow_probs