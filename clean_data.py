import re
from tqdm import tqdm

in_path = "./data/raw/weibo.tsv"
out_path = "./data/clean/clean_weibo.tsv"

remove = list('ノ¯︶ーσ･з∠~〜*&%$＊̀ ́；●ヾД≤≥ε┏゜ロ┛□▔﹏∇ψ❤三 ڡ ♂ㄒ;∂‸Ծˋ๑ºั`·ﾟゝ[]○▽￥←┴・｀.「\
                    ﹃『』」∩ヽ ﾟ∀ｏ`´╭╮【Σっ★╥¬☆＜⌒ﾉ】→↑°╰╯┴x•ㅂ…|\/^<>口︵—≧≦⊙ω∑√')

transmit = ["（转）", "（图转）", "「转」", "alink", "（转自网络）", "「图转」", "(转)", "(图片来自网络)"]

def filter_at(desstr, restr=''):
    try:
        res = re.compile(u'@([^\s|\/|:|@]+)')
    except re.error:
        print("error at filter_at")
    return res.sub(restr, desstr)

def filter_topic(desstr, restr=''):
    try:
        res = re.compile(u'#[^#]+#')
    except re.error:
        print("error at filter_at")
    return res.sub(restr, desstr)

def filter_emoji(desstr,restr=''):
    try:
        res = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        res = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    return res.sub(restr, desstr)


def handle_symbol(text):
    for r in remove:
        while r in text:
            text = text.replace(r,"")
    return text


def filter_transmit(text):
    for r in transmit:
        while r in text:
            text = text.replace(r, "")
    return text

with open(out_path, "w") as target:
    with open(in_path, 'r') as f:
        for line in tqdm(f):
            try:
                result = filter_at(line)
                result = filter_topic(result)
                result = filter_transmit(result)
                result = filter_emoji(result)
                result = handle_symbol(result)

                target.write(result)
            except:
                print("exception")
                continue
f.close()
target.close()