from transformers import BertTokenizer
from tqdm import tqdm

vocab_path = "./config/bert/vocab.txt"
train_path = "./data/stop.txt"
tokenizer = BertTokenizer(vocab_file=vocab_path)

ids = set()

with open(train_path, 'rb') as raw_f:
    data = raw_f.read().decode("utf-8")
raw_f.close()
train_data = data.split("\n")

for word in train_data:
    ids.add(tokenizer.convert_tokens_to_ids(word))
print(len(ids))
print(list(ids))