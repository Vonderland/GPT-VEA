import torch
import os
import random
import numpy as np
import argparse
import logging
from model import TransformerVAE
from transformers import BertTokenizer
from os.path import join, exists
from torch.nn import CrossEntropyLoss


from sklearn.model_selection import train_test_split
from tqdm import tqdm

logger = None
n_ctx = 300


def set_interact_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, required=False)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False)
    parser.add_argument('--vocab_path', default='vocabulary/vocab_small.txt', type=str, required=False)
    parser.add_argument('--train_tokenized_path', default='data/train_tokenized.txt', type=str,
                        required=False)
    parser.add_argument('--log_path', default='data/inference.log', type=str, required=False)
    parser.add_argument('--output_path', default='data/output.txt', type=str, required=False)
    parser.add_argument('--model_output_path', default='saved_model/', type=str, required=False)
    parser.add_argument('--decoder_config', default='pretrained/config.json', type=str, required=False)
    parser.add_argument('--topk', default=5, type=int, required=False)
    parser.add_argument('--topp', default=0, type=float, required=False)
    parser.add_argument('--repr_form', type=str, default="mean")
    parser.add_argument('--z_utilize', type=str, default="embedding")
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--without_bow', action='store_true')
    return parser.parse_args()


def set_random_seed(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_logger(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def create_model(args, vocab_size):
    """

    :param args:
    :param vocab_size:字典大小
    :return:
    """
    model = TransformerVAE(n_ctx=n_ctx, vocab_size=vocab_size, decoder_config=args.decoder_config, with_bow=(not args.without_bow),
                           z_utilize=args.z_utilize, repr_form=args.repr_form)
    return model

pad_idx = 0
unk_idx = 100
cls_idx = 101
sep_idx = 102

def get_text(tokenizer, ids):
    while pad_idx in ids:
        ids.remove(pad_idx)
    while unk_idx in ids:
        ids.remove(unk_idx)
    while cls_idx in ids:
        ids.remove(cls_idx)
    while sep_idx in ids:
        ids.remove(sep_idx)
    text = tokenizer.convert_ids_to_tokens(ids)
    return "".join(text)

def calculate_loss_and_accuracy(outputs, labels, device):

    # modified the gpt2 source codes
    logits = outputs[0]

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')  # ignore pad
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)

    not_ignore = shift_labels.ne(pad_id)
    num_targets = not_ignore.long().sum().item()

    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets
    return loss, accuracy


def main():
    global pad_id
    pad_id = 0

    global sep_id
    sep_id = 102

    args = set_interact_args()
    logger = create_logger(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    tokenizer = BertTokenizer(vocab_file=args.vocab_path)

    vocab_size = len(tokenizer)

    logger.info("loading data")
    with open(args.train_tokenized_path, "r", encoding="utf8") as f:
        data = f.read()
    data_list = data.split("\n")
    train_list, test_list = train_test_split(data_list, test_size=0.05, random_state=1)

    model = create_model(args, vocab_size)
    model.to(device)

    model_path = join(args.model_output_path, "saved.pt")
    if os.path.exists(model_path):
        if device == 'cpu':
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
    else:
        logger.info("no pretrained model!")

    model.eval()

    with open(args.output_path, "w", encoding="utf-8") as f:
        for test in tqdm(test_list):
            test = [int(i) for i in test.split()]
            input_tensor = torch.tensor(test).long().to(device)
            input_tensor = torch.unsqueeze(input_tensor, 0)
            gen, logits, mu, logvar = model.inference(input_tensor, device, args.max_len, args.topk, args.topp)

            ce, accuracy = calculate_loss_and_accuracy(logits, labels=input_tensor, device=device)

            kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()

            loss = ce + 0.5 * kld

            src = get_text(tokenizer, test)
            gen = get_text(tokenizer, gen)
            f.write("src:{}\n".format(src))
            f.write("gen:{}\n".format(gen))
            f.write("ce {:.6}, kld {:.6}, loss {:.6}, accuracy {:.6}\n".format(ce, kld, loss, accuracy))

if __name__ == '__main__':
    main()
