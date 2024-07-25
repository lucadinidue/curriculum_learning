from collections.abc import Callable
from classes import Token, Sentence
from complexity_functions import *
from tqdm import tqdm
import argparse
import csv

SAMPLE_SIZE = 10000000



def compute_sentence_complexities(dataset_path:str, complexity_function: Callable, split_clitics:bool=True, batch_size:int=256) -> list:
    sentences = []
    sentences_batch = []
    lines_to_skip = 0 # for words with clitics
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
            if line.strip() == '':
                sentences_batch.append(sentence)
                if len(sentences_batch) == batch_size:
                    complexity_function(sentences_batch)
                    sentences += sentences_batch
                    sentences_batch = []
        if sentences_batch:
            complexity_function(sentences_batch)
            sentences += sentences_batch
    return sentences



def write_sentences_to_file(sentences:list, out_path:str):
    with open(out_path, 'w') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        for sentence in sentences:
            csv_writer.writerow([sentence.sentence_id, sentence.text, sentence.complexity])

complexity_fuctions = {
    'sentence_length': {'function': compute_sentence_length, 'split_clitics': True},
    'perplexity': {'function': compute_model_perplexity, 'split_clitics': True},
    'gulpease': {'function': compute_gulpease_index, 'split_clitics': False}
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--sample_idx')
    parser.add_argument('-c', '--complexity_function', choices=['sentence_length', 'perplexity', 'gulpease'])
    args = parser.parse_args()

    conllu_path = f'data/dataset_samples/sample_{args.sample_idx}.conllu'
    out_path = f'data/dataset_samples/sample_{args.sample_idx}_{args.complexity_function}.tsv'
    
    sentences = compute_sentence_complexities(conllu_path, complexity_fuctions[args.complexity_function]['function'], complexity_fuctions[args.complexity_function]['split_clitics'])
    sorted_sentences = [sentence for sentence in sorted(sentences, key=lambda x: x.complexity)]
    write_sentences_to_file(sorted_sentences, out_path)


if __name__ == '__main__':
    main()



