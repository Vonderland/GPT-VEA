from transformers import BertTokenizer
from tqdm import tqdm

PAD = '[PAD]'
n_ctx = 300
vocab_path = "./vocabulary/vocab_small.txt"
train_path = "./data/clean/test.tsv"
tokenized_path = "./data/test.txt"

tokenizer = BertTokenizer(vocab_file=vocab_path)
vocab_size = len(tokenizer)

pad_id = tokenizer.convert_tokens_to_ids(PAD)
print("pad_id = ", pad_id)

print("tokenizing raw data,raw data path:{}, token output path:{}".format(train_path, tokenized_path))
with open(train_path, 'rb') as raw_f:
    data = raw_f.read().decode("utf-8")
raw_f.close()
train_data = data.split("\n")
print("there are {} dialogue in raw dataset".format(len(train_data)))

with open(tokenized_path, "w", encoding="utf-8") as f:
    for dialogue_index, dialogue in enumerate(tqdm(train_data)):
        utterances = dialogue.split("\t")
        # each dialog begins with [CLS]
        dialogue_ids = [tokenizer.cls_token_id]
        for utterance in utterances:
            dialogue_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in utterance])
            # each utterance ends with [SEP], standing for ending
            dialogue_ids.append(tokenizer.sep_token_id)
        # limit the length of each dialog
        dialogue_ids = dialogue_ids[:n_ctx]
        for dialogue_id in dialogue_ids:
            f.write(str(dialogue_id) + ' ')
        if dialogue_index < len(train_data) - 1:
            f.write("\n")
f.close()






