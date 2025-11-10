from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss
from utils import get_last_checkpoint
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import argparse
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_perplexity_old(sentences, tokenizer, model):
    encoding = tokenizer(
            sentences,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=False
        )

    encoded_texts = encoding['input_ids']
    attention_masks = encoding['attention_mask']

    perplexities = []
    loss_fct = CrossEntropyLoss(reduction='none')

    model.eval()

    for start_index in range(0, len(encoded_texts)):
        end_index = min(start_index + 1, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index].to(device)
        attention_mask = attention_masks[start_index:end_index].to(device)

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attention_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attention_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        perplexities += perplexity_batch.tolist()


    return perplexities


@torch.no_grad()
def compute_perplexity(sentences, tokenizer, model, max_length=128):
    encodings = tokenizer(
        sentences,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
        return_attention_mask=True,
        return_token_type_ids=False,
    )

    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    output = model(input_ids, attention_mask=attention_mask)
    logits = output.logits # (batch_size, num_tokens, vocab_size)

    # shift for next-token prediction
    shifted_logits = logits[..., :-1, :].contiguous()            # (batch_size, num_tokens-1, vocab_size)
    shifted_labels = input_ids[..., 1:].contiguous()             # (batch_size, num_tokens-1)
    shifted_mask   = attention_mask[..., 1:].contiguous()             # (batch_size, num_tokens-1)

    ignore_labels = shifted_labels.masked_fill(shifted_mask == 0, -100) # ignore padding tokens in loss computation

    token_negative_log_likelihood = F.cross_entropy(
        shifted_logits.transpose(1, 2),   # (batch_size, vocab_size, num_tokens-1)
        ignore_labels,                  # (batch_size, num_tokens-1)
        reduction='none'
    )


    num_non_padding_tokens = (ignore_labels != -100).sum(dim=1)            # (batch_size)
    sentence_negative_log_likelihood = token_negative_log_likelihood.sum(dim=1) / num_non_padding_tokens.clamp(min=1)
    sentence_negative_log_likelihood = sentence_negative_log_likelihood.masked_fill(num_non_padding_tokens == 0, float('nan'))


    perplexity = torch.exp(sentence_negative_log_likelihood)
    return perplexity.detach().cpu().tolist()




def compute_checkpoint_perplexity(model_path, tokenizer,  output_path, sentences_df, batch_size=32):
    model = AutoModelForCausalLM.from_pretrained(model_path).eval().to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    perplexities = []
    batched_sentences = []
    for sent_idx, row in tqdm(sentences_df.iterrows(), total=len(sentences_df)):
        batched_sentences.append(row['text'])
        if len(batched_sentences) == batch_size or sent_idx == len(sentences_df) - 1:
            batch_perplexities = compute_perplexity(batched_sentences, tokenizer, model)
            perplexities.extend(batch_perplexities)
            batched_sentences = []
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
    
    tokenizer_path = 'models/gpt_tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.output_dir is None:
        args.output_dir = os.path.join('data/perplexity/gpt', args.dataset, args.model_path.split('/')[-1])
    
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