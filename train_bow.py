import transformers
import torch
import os
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.nn import DataParallel
import logging
from model import TransformerBOW
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
    parser.add_argument('--device', default='1', type=str, required=False)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--vocab_path', default='vocabulary/vocab_small.txt', type=str, required=False)
    parser.add_argument('--train_tokenized_path', default='data/train_tokenized.txt', type=str,
                        required=False)
    parser.add_argument('--log_path', default='data/bow_training.log', type=str, required=False)
    parser.add_argument('--epochs', default=1, type=int, required=False)
    parser.add_argument('--batch_size', default=16, type=int, required=False)
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False)
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False)
    parser.add_argument('--log_step', default=1, type=int, required=False)
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--model_output_path', default='tbow_saved_model/', type=str, required=False)
    parser.add_argument('--writer_dir', default='tbow_tensorboard_summary/', type=str, required=False)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--save_step_percentage', type=float, default=0.05)
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


def create_model(vocab_size):

    model = TransformerBOW(n_ctx=n_ctx, vocab_size=vocab_size)
    return model


def calculate_bow(bow_probs, input_ids, device):
    bow = 0.0
    if bow_probs is not None:
        # skip the [cls]
        label = input_ids[:, 1:].contiguous().to(device)
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


def train(model, device, train_list, multi_gpu, args):
    train_dataset = MyDataset(train_list)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                  collate_fn=collate_fn)

    total_steps = int(train_dataset.__len__() * args.epochs / args.batch_size / args.gradient_accumulation)
    logger.info('total training steps = {}'.format(total_steps))

    save_step = max(int(args.save_step_percentage * total_steps), 1)
    logger.info('save per {} steps'.format(save_step))

    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=total_steps)

    logger.info('starting training')
    running_loss = 0
    overall_step = -1


    model_path = join(args.model_output_path, "saved.pt")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        running_loss = checkpoint['running_loss']
        overall_step = checkpoint['overall_step']
    logger.info("running loss:{}, overall step:{}".format(running_loss, overall_step))
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    oom_time = 0
    model.train()
    oom_flag = False
    epoch_start_time = datetime.now()
    for batch_idx, input_ids in enumerate(train_dataloader):
        if batch_idx <= overall_step:
            continue

        input_ids = input_ids.to(device)
        try:
            mu, logvar, bow_probs = model.forward(input=input_ids)

            bow_loss = calculate_bow(bow_probs, input_ids, device)

            loss = bow_loss

            if multi_gpu:
                loss = loss.mean()
            if args.gradient_accumulation > 1:
                loss = loss / args.gradient_accumulation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if (batch_idx + 1) % args.gradient_accumulation == 0:
                running_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                overall_step += 1
                if (overall_step + 1) % args.log_step == 0 or (overall_step + 1 == total_steps):
                    logger.info(
                        "step {}, loss {:.6}".format(overall_step, loss))
                    tb_writer.add_scalar('loss', loss.item(), overall_step)
            if (overall_step + 1) % save_step == 0 or (overall_step == total_steps):
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


def evaluate(model, device, test_list, multi_gpu, args):
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
            input_ids.to(device)
            mu, logvar, bow_probs = model.forward(input=input_ids)
            bow_loss = calculate_bow(bow_probs, input_ids,device)

            loss = bow_loss

            if multi_gpu:
                loss = loss.mean()
            if args.gradient_accumulation > 1:
                loss = loss / args.gradient_accumulation
            logger.info("evaluate batch {}, loss {:.6}".format(batch_idx, loss))
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

    model = create_model(vocab_size)
    model.to(device)

    multi_gpu = False

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
    train(model, device, train_list, multi_gpu, args)
    evaluate(model, device, test_list, multi_gpu, args)


if __name__ == '__main__':
    main()
