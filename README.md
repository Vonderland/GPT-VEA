# GPT-VEA
Transformer-based VAE network with GPT2 as the decoder, and I have tried this architecture on dialog response generation task.

## dataset
I used the weibo comment dataset(https://github.com/codemayq/chinese_chatbot_corpus) as the corpus.

## file description
- model folder: it contains several parts. In nn folder, there are two adapted GPT2, using cls or embedding to inject latent z. transformer_vae.py implements the architecture of the model and transformer_bow.py only contains components of encoder and the bow part to learn details of bow.
- vocabulary folder: it contains the vocab file for the tokenizer.
- clean_data.py: 
