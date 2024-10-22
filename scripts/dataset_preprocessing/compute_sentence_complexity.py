from transformers import AutoModelForCausalLM, AutoTokenizer
from collections.abc import Callable
from classes import Token, Sentence
from complexity_functions import *
from tqdm import tqdm
import argparse
import csv
import os

SAMPLE_SIZE = 10000000
ORDERED_KEYS = ['base', 'lexical', 'syntax', 'all']

def compute_sentence_complexities(dataset_path:str, out_path:str, complexity_function: Callable, split_clitics:bool=True, batch_size:int=16, last_id:int=None, **kwargs) -> list:
    sentences = []
    sentences_batch = []
    lines_to_skip = 0 # for words with clitics
    found_last_id = True if last_id is None else False
    
    with tqdm(total=SAMPLE_SIZE) as pbar:
        for line in open(dataset_path):
            if line.startswith('# sent_id = '):
                pbar.update(1)
                sent_id = line[len('# sent_id = '):].strip()
                if sent_id == last_id:
                    found_last_id = True
                sentence = Sentence(sent_id)
            elif not found_last_id:
                continue
            elif line.startswith('# text = '):
                text = line[len('# text = '):].strip()
                sentence.set_text(text)
            elif line[0].isdigit():
                token = Token(line)
                if '-' in token.token_id:
                    if not split_clitics:
                        ids_to_skip = line.split('\t')[0].split('-')
                        lines_to_skip = int(ids_to_skip[1]) - int(ids_to_skip[0]) + 1 # number of lines to skip
                        take_linguistic_features = True # we take the linguistic annotation of the next word
                        sentence.add_token(token)
                else:
                    if lines_to_skip == 0:
                        sentence.add_token(token)
                    elif take_linguistic_features:
                        sentence.tokens[-1].override_linguistic_features(line)
                        take_linguistic_features = False
                    lines_to_skip = max(0, lines_to_skip-1)
            elif line.strip() == '':
                if found_last_id and sent_id != last_id:
                    sentences_batch.append(sentence)
                if len(sentences_batch) == batch_size:
                    complexity_function(sentences_batch, **kwargs)
                    sentences += sentences_batch
                    sentences_batch = []
                if len(sentences) >= 1000:
                    write_sentences_to_file(sentences, out_path)
                    sentences = []

        if len(sentences_batch) > 0:
            complexity_function(sentences_batch, **kwargs)
            sentences += sentences_batch
            write_sentences_to_file(sentences, out_path)


def write_sentences_to_file(sentences:list, out_path:str):
    with open(out_path, 'a') as out_file:
        csv_writer = csv.writer(out_file, delimiter=',')
        for sentence in sentences:
            if type(sentence.complexity) == dict:
                complexities = [sentence.complexity[key] for key in ORDERED_KEYS]
                csv_writer.writerow([sentence.sentence_id, sentence.text] + complexities)
            else:    
                csv_writer.writerow([sentence.sentence_id, sentence.text, sentence.complexity])


def instantiate_model_and_tokenizer(model_name:str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


complexity_functions = {
    'sentence_length': {'function': compute_sentence_length, 'split_clitics': True},
    'perplexity': {'function': compute_model_perplexity, 'split_clitics': True},
    'gulpease': {'function': compute_gulpease_index, 'split_clitics': False},
    'readit': {'function': compute_readit_score, 'split_clitics': True}
}


def load_last_index(src_path):
    last_id = None
    with open(src_path, 'r') as src_file:
        csv_reader = csv.reader(src_file, delimiter=',')
        for line in csv_reader:
            last_id = line[0]
    return last_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--sample_idx')
    parser.add_argument('-c', '--complexity_function', choices=['sentence_length', 'perplexity', 'gulpease', 'readit'])
    parser.add_argument('-m', '--model_name', default='local_models/Minerva-350M-base-v1.0')
    parser.add_argument('-b', '--batch_size', type=int, default=1024)
    parser.add_argument('-r', '--restart', action='store_true')
    args = parser.parse_args()

    conllu_path = f'data/dataset_samples/sample_{args.sample_idx}.conllu'
    out_path = f'data/dataset_samples/sample_{args.sample_idx}_{args.complexity_function}.csv'

    last_id = None
    if os.path.exists(out_path):
        if args.restart:
            os.remove(out_path)
            last_id = None
        else:
            last_id = load_last_index(out_path)

    print(f'Starting from id = {last_id}.')

    kwargs = {}
    if args.complexity_function == 'perplexity':
        model, tokenizer = instantiate_model_and_tokenizer(args.model_name)
        kwargs = {'model': model, 'tokenizer': tokenizer}

    sentences = compute_sentence_complexities(conllu_path, out_path, complexity_functions[args.complexity_function]['function'], complexity_functions[args.complexity_function]['split_clitics'], batch_size=args.batch_size, last_id=last_id, **kwargs)
    sorted_sentences = sentences
    write_sentences_to_file(sorted_sentences, out_path)


if __name__ == '__main__':
    main()



