from transformers import AutoModelForMaskedLM, AutoTokenizer
from utils import get_last_checkpoint
from tqdm import tqdm
import pandas as pd
import argparse
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mask_all_tokens(sentence, tokenizer):
    masked_sentences = []
    tokenized_sentence = tokenizer(sentence, return_tensors='pt')
    for token_idx in range(1, len(tokenized_sentence['input_ids'][0]) - 1):
        masked_input = tokenizer(sentence, return_tensors='pt')  # tokenized_sentence.copy()
        masked_input['input_ids'][0][token_idx] = tokenizer.mask_token_id
        masked_sentences.append(masked_input)
    return masked_sentences

def compute_plausibility_and_perplexity(sentence, tokenizer, model):
    masked_sentences = mask_all_tokens(sentence, tokenizer)
    original_tokens = tokenizer(sentence)['input_ids']

    masked_probabilities = []
    losses = []

    for sent_idx, masked_sentence in enumerate(masked_sentences):
        masked_idx = sent_idx + 1
        correct_token = original_tokens[masked_idx]
        labels = torch.full_like(masked_sentence['input_ids'], -100)
        labels[0, masked_idx] = correct_token


        with torch.no_grad():
            outputs = model(input_ids=masked_sentence['input_ids'].to(device), attention_mask=masked_sentence['attention_mask'].to(device), labels=labels.to(device))

        logits = outputs.logits[0, masked_idx]
        probs = logits.softmax(dim=-1)
        
        losses.append(outputs.loss)
        masked_probabilities.append(probs[correct_token].item())

    plausibility = sum(masked_probabilities) / len(masked_probabilities)
    perplexity = torch.exp(torch.stack(losses).mean()).item()

    return plausibility, perplexity


def compute_checkpoint_perplexity(model_path, tokenizer,  output_path, sentences_df):
    model = AutoModelForMaskedLM.from_pretrained(model_path).eval().to(device)

    plausibilities, perplexities = [], []
    for _, row in tqdm(sentences_df.iterrows(), total=len(sentences_df)):
        sent_plausibility, sent_perplexity = compute_plausibility_and_perplexity(row['sent_id'], tokenizer, model)
        plausibilities.append(sent_plausibility)
        perplexities.append(sent_perplexity)
    
    sentences_df['plausibility'] = plausibilities
    sentences_df['perplexity'] = perplexities

    sentences_df.to_csv(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-n', '--num_sentences', type=int, default=1000)
    parser.add_argument('-d', '--dataset', type=str, choices=['wikipedia', 'treebank'])
    parser.add_argument('-o', '--output_dir', type=str, default=None)
    args = parser.parse_args()

    if args.dataset == 'wikipedia':
        dataset_path = 'data/datasets/eval_3.csv'
        sentences_df = pd.read_csv(dataset_path).head(args.num_sentences)[['sent_id', 'text']]
    else:
        dataset_path = 'data/probing_data/test.tsv'
        sentences_df = pd.read_csv(dataset_path, sep='\t').head(args.num_sentences)[['identifier', 'text']]
        sentences_df.rename(columns={'identifier': 'sent_id'}, inplace=True)
    
    tokenizer_path = 'models/bert_tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    if args.output_dir is None:
        args.output_dir = os.path.join('data/perplexity', args.dataset, args.model_path.split('/')[-1])
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    last_training_step = get_last_checkpoint(args.model_path)

    for checkpoint_name in os.listdir(args.model_path):
        checkpoint_model_path = os.path.join(args.model_path, checkpoint_name)
        if os.path.isdir(checkpoint_model_path):
            output_path = os.path.join(args.output_dir, checkpoint_name+'.csv')
            compute_checkpoint_perplexity(checkpoint_model_path, tokenizer,  output_path, sentences_df)
    
    compute_checkpoint_perplexity(args.model_path, tokenizer, os.path.join(args.output_dir, f'checkpoint-{last_training_step}'), sentences_df)

if __name__ == '__main__':
    main()


        