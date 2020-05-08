import transformers
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
    parser.add_argument('--decoder_config', default='pretrained/config.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--repr_form', type=str, default="mean", help="z的表示方式")
    parser.add_argument('--z_utilize', type=str, default="embedding", help="z的使用方式")
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
    model = TransformerVAE(n_ctx=n_ctx, vocab_size=vocab_size, decoder_config=args.decoder_config,
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
    """
    计算非pad_id的平均loss和准确率
    :param outputs:
    :param labels:
    :param device:
    :return:
    """
    logits = outputs[0]  # 每个token用来预测下一个token的prediction_score,维度:[batch_size,token_len,voca_size]
    # 用前n-1个token，预测出第n个token
    # 用第i个token的prediction_score用来预测第i+1个token。
    # 假定有input有n个token，则shift_logits表示model中第[0,n-2]个token的prediction_score，shift_labels表示第[1，n-1]的label
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction='sum')  # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]

    # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
    not_ignore = shift_labels.ne(pad_id)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
    num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

    correct = (shift_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的token
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

    flag = False
    with open(args.output_path, "w", encoding="utf-8") as f:
        for test in tqdm(test_list):
            test = [int(i) for i in test.split()]
            input_tensor = torch.tensor(test).long().to(device)
            input_tensor = torch.unsqueeze(input_tensor, 0)
            if not flag:
                print(input_tensor.shape)
                flag = True
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
