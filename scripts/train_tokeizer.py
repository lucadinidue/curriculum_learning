from tokenizers import BertWordPieceTokenizer
import json
import os

def main():
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


if __name__ == '__main__':
    main()