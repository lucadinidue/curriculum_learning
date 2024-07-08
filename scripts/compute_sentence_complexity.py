from collections.abc import Callable
from classes import Token, Sentence
from complexity_functions import *
from tqdm import tqdm
import argparse
import csv
import os

SAMPLE_SIZE = 10000000



def compute_sentence_complexities(dataset_path:str, complexity_function: Callable, split_clitics:bool=True, batch_size:int=16) -> list:
    sentences = []
    sentences_batch = []
    with tqdm(total=SAMPLE_SIZE) as pbar:
        for line in open(dataset_path):
            if line.startswith('# sent_id = '):
                pbar.update(1)
                sent_id = line[len('# sent_id = '):].strip()
                sentence = Sentence(sent_id)
            if line.startswith('# text = '):
                text = line[len('# text = '):].strip()
                sentence.set_text(text)
            if line[0].isdigit():
                token_id = line.split('\t')[0]
                if split_clitics:
                    if '-' not in token_id:
                        sentence.add_token(line.strip())
                else:
                    raise Exception('Not implemented yet')         
            if line.strip() == '':
                sentences_batch.append(sentence)
                if len(sentences_batch) == batch_size:
                    complexity_function(sentences_batch)
                    sentences += sentences_batch
                    sentence_batch = []
        if sentence_batch:
            complexity_function(sentences_batch)
            sentences += sentences_batch
    return sentences



def write_sentences_to_file(sentences:list, out_path:str):
    with open(out_path, 'w') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        for sentence in sentences:
            csv_writer.writerow(sentence.text)

complexity_fuctions = {
    'sentence_length': compute_sentence_length,
    'perplexity': compute_model_perplexity
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--sample_idx')
    parser.add_argument('-c', '--complexity_function', choices=['sentence_length', 'perplexity'])
    args = parser.parse_args()

    conllu_path = f'data/dataset_samples_10M/sample_{args.sample_idx}.conllu'
    out_path = f'data/dataset_samples_10M/sample_{args.sample_idx}_{args.complexity_function}.tsv'
    
    sentences = compute_sentence_complexities(conllu_path, complexity_fuctions[args.complexity_function])
    sorted_sentences = [sentence for sentence in sorted(sentences, key=lambda x: x.complexity)]
    write_sentences_to_file(sorted_sentences, out_path)


if __name__ == '__main__':
    main()



