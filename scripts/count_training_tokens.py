from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd
import argparse
import json
import os


# def load_checkpoint_nums(curriculum):
#     checkpoints = []
#     checkpoints_dir = f'models/pretrained/bert/bert_medium_42_train_1_{curriculum}' # all models have the same checkpoints
#     for checkpoint_subdir in os.listdir(checkpoints_dir):
#         if checkpoint_subdir.startswith('checkpoint-'):
#             checkpoint_num = int(checkpoint_subdir.split('-')[1])
#             checkpoints.append(checkpoint_num)
#     return sorted(checkpoints)

def load_and_tokenize_dataset(dataset_path, tokenizer, text_column_name='text', max_seq_length=128, padding=False):
    df = pd.read_csv(dataset_path)
    data = Dataset.from_pandas(df)

    def tokenize_function(examples):
        # Remove empty lines
        examples[text_column_name] = [
            line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )

    tokenized_datasets = data.map(
            tokenize_function,
            batched=True,
            remove_columns=[text_column_name],
    )
    return tokenized_datasets

def count_tokens(tokenized_dataset, sentences_to_checkpoints):
    ckeckpoints_to_tokens = {}
    num_tokens = 0
    for epoch in range(1,4):
        for sent_idx, sentence_tokens in enumerate(tokenized_dataset['input_ids']):
            num_tokens += len(sentence_tokens) -2 # remove [CLS] and [SEP]
            sent_num = (sent_idx + 1) + (epoch-1) * len(tokenized_dataset)
            if sent_num in sentences_to_checkpoints:
                ckeckpoints_to_tokens[sentences_to_checkpoints[sent_num]] = num_tokens
    return ckeckpoints_to_tokens


def load_tokenizer(model_type):
    if model_type == 'bert':
        tokenizer_path = 'models/bert_tokenizer'
    else:
        tokenizer_path = 'models/gpt_tokenizer'
    return AutoTokenizer.from_pretrained(tokenizer_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--curriculum', type=str, required=True)
    parser.add_argument('-m', '--model_type', type=str, choices=['bert', 'gpt'])
    args = parser.parse_args()

    batch_size = 128 * 2 # per device batch size * GPUs
    tokenizer = load_tokenizer(args.model_type)

    
    dataset_path = f'data/datasets/train_1_{args.curriculum}.csv'    
    # checkpoint_list = load_checkpoint_nums(args.curriculum)
    checkpoint_list = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000, 8000, 12000, 16000, 
                       20000, 24000, 28000, 32000, 36000, 40000, 44000, 48000, 52000, 56000, 60000, 
                       64000, 68000, 72000, 76000, 80000, 84000, 88000, 92000, 96000, 100000, 104000, 
                       108000, 112000, 116000]

 

    checkpoints_to_sentences = {checkpoint_num: checkpoint_num * batch_size for checkpoint_num in checkpoint_list}
    sentences_to_checkpoints = {value: key for key, value in checkpoints_to_sentences.items()}

    tokenized_datasets = load_and_tokenize_dataset(dataset_path, tokenizer)
    checkpoints_tokens = count_tokens(tokenized_datasets, sentences_to_checkpoints)

    with open(os.path.join(f'data/num_training_tokens/{args.model_type}', f'{args.curriculum}.json'), 'w') as f:
        json.dump(checkpoints_tokens, f)

if __name__ == '__main__':
    main()
