import transformers
import torch
import os
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import logging
from model import TransformerVAE
from transformers import BertTokenizer
from os.path import join, exists
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split

logger = None
n_ctx = 300


def setup_train_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--decoder_config', default='pretrained/config.json', type=str, required=False)
    parser.add_argument('--vocab_path', default='vocabulary/vocab_small.txt', type=str, required=False)
    parser.add_argument('--train_tokenized_path', default='data/train_tokenized.txt', type=str,
                        required=False)
    parser.add_argument('--log_path', default='data/training.log', type=str, required=False)
    parser.add_argument('--epochs', default=1, type=int, required=False)
    parser.add_argument('--batch_size', default=16, type=int, required=False)
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False)
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False)
    parser.add_argument('--log_step', default=50, type=int, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--model_output_path', default='saved_model/', type=str, required=False)
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--kl_anneal_function', type=str, default='logistic')
    parser.add_argument('--kl_anneal_percentage', type=float, default=0.15)
    parser.add_argument('--kl_anneal_k', type=float, default=0.00025)
    parser.add_argument('--save_step_percentage', type=float, default=0.05)
    parser.add_argument('--word_dropout', type=float, default=0)
    parser.add_argument('--without_bow', action='store_true')
    parser.add_argument('--repr_form', type=str, default="mean")
    parser.add_argument('--z_utilize', type=str, default="embedding")
    parser.add_argument('--bow_weight', type=float, default=5.0)

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

    model = TransformerVAE(n_ctx=n_ctx, vocab_size=vocab_size, decoder_config=args.decoder_config,
                           word_dropout=args.word_dropout, with_bow=(not args.without_bow),
                           z_utilize=args.z_utilize, repr_form=args.repr_form)
    return model


def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'linear':
        return min(1, step / x0)


def get_bow_weight(step, k, x0):
    return 1 + 9 * (1 - float(1 / (1 + np.exp(-k * (step - x0)))))



def calculate_loss_and_accuracy(outputs, labels, device):

    logits = outputs[0]

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
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


def calculate_bow(bow_probs, input_ids, device):
    bow = 0.0
    if bow_probs is not None:
        # skip the [cls]
        label = input_ids[:, 1:].contiguous().to(device)
        label[label == sep_id] = pad_id

        bow_probs = bow_probs.unsqueeze(1)
        bow_probs = bow_probs.expand(bow_probs.shape[0], label.shape[1], bow_probs.shape[2]).contiguous()

        loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
        bow = loss_fct(bow_probs.view(-1, bow_probs.size(-1)),
                       label.view(-1))

        not_ignore = label.ne(pad_id)
        num_targets = not_ignore.long().sum().item()
        bow = bow / num_targets
    return bow


def collate_fn(batch):
    global pad_id
    input_ids = []
    btc_size = len(batch)
    max_input_len = 0

    for btc_idx in range(btc_size):
        if max_input_len < len(batch[btc_idx]):
            max_input_len = len(batch[btc_idx])

    for btc_idx in range(btc_size):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([pad_id] * (max_input_len - input_len))
    return torch.tensor(input_ids, dtype=torch.long)


def train(model, device, train_list, args):
    train_dataset = MyDataset(train_list)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                  collate_fn=collate_fn)

    total_steps = int(train_dataset.__len__() * args.epochs / args.batch_size)
    logger.info('total training steps = {}'.format(total_steps))

    save_step = max(int(args.save_step_percentage * total_steps), 1)
    logger.info('save per {} steps'.format(save_step))

    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=total_steps)

    logger.info('starting training')
    running_loss = 0
    overall_step = -1
    kl_anneal_x0 = int(total_steps * args.kl_anneal_percentage)

    model_path = join(args.model_output_path, "saved.pt")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # finished_epoch = checkpoint['finished_epoch'] + 1
        running_loss = checkpoint['running_loss']
        overall_step = checkpoint['overall_step']
    logger.info("running loss:{}, overall step:{}".format(running_loss, overall_step))
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    oom_time = 0
    model.train()
    oom_flag = False
    # for epoch in range(finished_epoch, args.epochs):
    epoch_start_time = datetime.now()
    for batch_idx, input_ids in enumerate(train_dataloader):
        if batch_idx <= overall_step:
            continue

        input_ids = input_ids.to(device)
        try:
            outputs, mu, logvar, bow_probs = model.forward(input=input_ids)
            # anneal_function, step, k, x0
            ce, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids, device=device)

            kl_weight = min(0.5, kl_anneal_function(anneal_function=args.kl_anneal_function, step=overall_step,
                                                       k=args.kl_anneal_k, x0=kl_anneal_x0))
            kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()


            bow_loss = calculate_bow(bow_probs, input_ids, device)

            loss = ce + kl_weight * kld + args.bow_weight * bow_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            running_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            overall_step += 1
            if overall_step == 0 or (overall_step + 1) % args.log_step == 0 or (overall_step + 1 == total_steps):
                logger.info(
                    "step {}, ce {:.6}, kld {:.6}, kl_weight {:.6}, bow {:.6}, bow_weight {:.6}, loss {:.6}, accuracy {:.6}".format(overall_step, ce, kld, kl_weight, bow_loss, args.bow_weight, loss, accuracy))
                tb_writer.add_scalar('ce', ce.item(), overall_step)
                tb_writer.add_scalar('kld', kld.item(), overall_step)
                tb_writer.add_scalar('loss', loss.item(), overall_step)
            if (overall_step + 1) % save_step == 0 or (overall_step + 1 == total_steps):
                logger.info('saving for step {}'.format(overall_step))
                if not os.path.exists(args.model_output_path):
                    os.mkdir(args.model_output_path)

                torch.save({
                    # 'finished_epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'overall_step': overall_step,
                    'running_loss': running_loss
                }, model_path)

                decoder_path = join(args.model_output_path, 'decoder/')

                if not os.path.exists(decoder_path):
                    os.mkdir(decoder_path)

                model.save_decoder(decoder_path)
                logger.info('finish saving for step {}'.format(overall_step))

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                oom_time += 1
                if not oom_flag:
                    logger.info("WARNING: ran out of memory,times: {}".format(oom_time))
                    logger.info("batch_idx = ", batch_idx)
                    oom_flag = True
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    epoch_finish_time = datetime.now()
    logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))
    logger.info('training finished')


def evaluate(model, device, test_list, args):
    model_path = join(args.model_output_path, "saved.pt")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
    logger.info("start evaluating model")
    model.eval()
    logger.info('starting evaluating')
    test_dataset = MyDataset(test_list)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                 collate_fn=collate_fn)
    with torch.no_grad():
        for batch_idx, input_ids in enumerate(test_dataloader):
            input_ids = input_ids.to(device)
            outputs, mu, logvar, bow_probs = model.forward(input=input_ids)
            # anneal_function, step, k, x0
            ce, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids, device=device)
            kld = (-0.5 * torch.sum(logvar - torch.pow(mu, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()

            bow_loss = calculate_bow(bow_probs, input_ids, device)

            loss = ce + 0.5 * kld + args.bow_weight * bow_loss

            logger.info("evaluate batch {}, ce {:.6}, kld {:.6}, bow {:.6}, loss {:.6}, accuracy {:.6}".format(batch_idx, ce, kld, bow_loss, loss, accuracy))
        logger.info("finishing evaluating")


def main():
    args = setup_train_args()
    global logger
    logger = create_logger(args)
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))

    if args.seed:
        set_random_seed(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    vocab_size = len(tokenizer)

    global pad_id
    pad_id = 0

    global sep_id
    sep_id = 102

    model = create_model(args, vocab_size)
    model.to(device)

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))

    logger.info("loading training data")
    with open(args.train_tokenized_path, "r", encoding="utf8") as f:
        data = f.read()
    data_list = data.split("\n")
    train_list, test_list = train_test_split(data_list, test_size=0.05, random_state=1)
    train(model, device, train_list, args)
    evaluate(model, device, test_list, args)


if __name__ == '__main__':
    main()
