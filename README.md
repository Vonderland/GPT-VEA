# GPT-VEA
Transformer-based VAE network with GPT2 as the decoder, and I have tried this architecture on dialog response generation task.

## dataset
I used the weibo comment dataset(https://github.com/codemayq/chinese_chatbot_corpus) as the corpus.

## file description
- model folder: it contains several parts. In nn folder, there are two adapted GPT2, using cls or embedding to inject latent z. transformer_vae.py implements the architecture of the model and transformer_bow.py only contains components of encoder and the bow part to learn details of bow.
- vocabulary folder: it contains the vocab file for the tokenizer.
- clean_data.py: do some data cleansing job on the weibo dataset, like removing the emoji, topic hash tag, the symbol "@" and so on.
- dataset.py: dataset for loading data during training.
- get_stop_ids.py: get ids for stop words, using in bow calculation.
- inference.py: after training, run it to generate new samples. 
- tokenize_data.py: use BertTokenizer to tokenize the dataset, based on vocab file in the vocabulary folder.
- train.py: run it to train the GPT-VAE model
- train.py: just train the encoder and the bow part, without decoder part to learn details of bow.
