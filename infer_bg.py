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
from model import BGVAE
from transformers import BertTokenizer
from os.path import join, exists
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logger = None
n_ctx = 300


def set_interact_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--vocab_path', default='vocabulary/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--train_tokenized_path', default='data/train_tokenized.txt', type=str,
                        required=False,
                        help='将原始训练语料tokenize之后的数据的存放位置')
    parser.add_argument('--log_path', default='data/inference.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--output_path', default='data/output.txt', type=str, required=False, help='测试语料生成存放位置')
    parser.add_argument('--model_output_path', default='saved_model/', type=str, required=False,
                        help='对话模型输出路径')
    parser.add_argument('--pretrained_decoder', default='saved_model/decoder', type=str, required=False, help='预训练的GPT2模型的路径')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--max_len', type=int, default=100, help='每个utterance的最大长度,超过指定长度则进行截断')
    return parser.parse_args()


def set_random_seed(args):
    """
    设置训练的随机种子
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
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
    model = BGVAE(vocab_size=vocab_size, decoder_config=args.decoder_config, repr_form=args.repr_form)
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

def main():
    args = set_interact_args()
    logger = create_logger(args)
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # 初始化tokenizer
    tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    # tokenizer的字典大小
    vocab_size = len(tokenizer)
    # 加载数据
    logger.info("loading data")
    with open(args.train_tokenized_path, "r", encoding="utf8") as f:
        data = f.read()
    data_list = data.split("\n")
    train_list, test_list = train_test_split(data_list, test_size=0.001, random_state=1)

    model = create_model(args, vocab_size)
    model.to(device)

    model_path = join(args.model_output_path, "saved.pt")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
    else:
        logger.info("no pretrained model!")

    model.eval()

    flag = False
    with open(args.output_path, "w", encoding="utf-8") as f:
        for test in tqdm(test_list):
            test = [int(i) for i in test.split()]
            input_tensor = torch.tensor(test).long().to(device)
            input_tensor = torch.unsqueeze(input_tensor, 0)
            if not flag:
                print(input_tensor.shape)
                flag = True
            gen = model.inference(input_tensor, device, args.max_len, args.topk, args.topp)
            src = get_text(tokenizer, test)
            gen = get_text(tokenizer, gen)
            f.write("src:{}\n".format(src))
            f.write("gen:{}\n".format(gen))

if __name__ == '__main__':
    main()
