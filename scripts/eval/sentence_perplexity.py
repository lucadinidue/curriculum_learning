from transformers import AutoModelForMaskedLM, AutoTokenizer
from utils import get_last_checkpoint
from tqdm import tqdm
import pandas as pd
import argparse
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mask_all_tokens(sentence, tokenizer):
    tokenized_sentence = tokenizer(sentence, return_tensors='pt')
    num_tokens = tokenized_sentence['input_ids'].shape[1]

    masking_positions = torch.arange(1, num_tokens-1, dtype=torch.long)  # (seq_len) - 2 for [CLS] and [SEP]
    masked_input_ids = tokenized_sentence['input_ids'].repeat(len(masking_positions), 1)     # (seq_len - 2, seq_len )
    masked_input_ids[torch.arange(len(masking_positions)), masking_positions] = tokenizer.mask_token_id   # (seq_len - 2, seq_len )

    return masked_input_ids, tokenized_sentence['input_ids'][0]


def compute_plausibility_and_perplexity(sentence, tokenizer, model, batch_size=128):
    masked_sentences, original_tokens = mask_all_tokens(sentence, tokenizer)
    masked_probabilities = []
    losses = []

    for batch_start in range(0, len(masked_sentences), batch_size):
        batch_sentences = masked_sentences[batch_start:batch_start+batch_size]  # batch_size, seq_len
        masked_ids = list(range(batch_start+1, min(batch_start+batch_size, len(masked_sentences))+1)) # batch_size
        correct_tokens = original_tokens[masked_ids]    # batch_size
        labels = torch.full_like(batch_sentences, -100)  # batch_size, seq_len

        batch_idx = torch.arange(len(batch_sentences))  # batch_size
        labels[batch_idx, masked_ids] = correct_tokens  # batch_size, seq_len

        with torch.no_grad():
            outputs = model(input_ids=batch_sentences.to(device), labels=labels.to(device))

        logits = outputs.logits[batch_idx, masked_ids] # batch_size, vocab_size
        probs = logits.softmax(dim=-1) # batch_size, vocab_size
        losses.append(outputs.loss)
        correct_tokens_probs = probs[batch_idx, correct_tokens] # batch_size
        masked_probabilities.extend(correct_tokens_probs.tolist())


    plausibility = sum(masked_probabilities) / len(masked_probabilities)
    perplexity = torch.exp(torch.stack(losses).mean()).item()

    return plausibility, perplexity


def compute_checkpoint_perplexity(model_path, tokenizer,  output_path, sentences_df):
    model = AutoModelForMaskedLM.from_pretrained(model_path).eval().to(device)

    plausibilities, perplexities = [], []
    for _, row in tqdm(sentences_df.iterrows(), total=len(sentences_df)):
        sent_plausibility, sent_perplexity = compute_plausibility_and_perplexity(row['text'], tokenizer, model)
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
    
    compute_checkpoint_perplexity(args.model_path, tokenizer, os.path.join(args.output_dir, f'checkpoint-{last_training_step}.csv'), sentences_df)

if __name__ == '__main__':
    main()


        