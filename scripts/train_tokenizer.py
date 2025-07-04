from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast
import argparse
import json
import os

def train_bert_tokenizer():
    dataset_path = '/home/luca/Workspace/wiki_ita/wiki_sentences.txt'
    tokenizer_path = 'models/bert_tokenizer'


    tokenizer_config = {
        "do_lower_case": "false",
        "strip_accents": "false"
    }

    with open(os.path.join(tokenizer_path, 'tokenizer_config.json'), 'w') as config_file:
        json.dump(tokenizer_config, config_file)

    tokenizer = BertWordPieceTokenizer(
        clean_text=True, 
        handle_chinese_chars=True, 
        strip_accents=False, 
        lowercase=False
    )

    tokenizer.train(files=[dataset_path],
        vocab_size=30000,
        min_frequency=2
    )

    tokenizer.save_model(tokenizer_path)

def train_gpt_tokenizer():
    dataset_path = '/home/luca/Workspace/wiki_ita/wiki_sentences.txt'
    tokenizer_path = 'models/gpt_tokenizer'

    tokenizer = ByteLevelBPETokenizer(lowercase=False)

    tokenizer.train(files=[dataset_path],
        vocab_size=30000,
        min_frequency=2,
        special_tokens=['<|endoftext|>'] 
    )

    tokenizer.save_model(tokenizer_path)

    gpt2_tokenizer = GPT2TokenizerFast(
        vocab_file = os.path.join(tokenizer_path, 'vocab.json'),
        merges_file = os.path.join(tokenizer_path, 'merges.txt')
    )

    gpt2_tokenizer.add_special_tokens({
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>"  # Se vuoi anche il pad token, opzionale
    })

    gpt2_tokenizer.model_max_length = 512
    gpt2_tokenizer.save_pretrained(tokenizer_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, choices=['bert', 'gpt'])
    args = parser.parse_args()

    if args.type == 'bert':
        train_bert_tokenizer()
    elif args.type == 'gpt':
        train_gpt_tokenizer()
    else:
        raise ValueError("Invalid tokenizer type. Choose 'bert' or 'gpt'.")
    
if __name__ == '__main__':
    main()