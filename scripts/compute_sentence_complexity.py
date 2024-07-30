from transformers import AutoModelForCausalLM, AutoTokenizer
from collections.abc import Callable
from classes import Token, Sentence
from complexity_functions import *
from tqdm import tqdm
import argparse
import csv

SAMPLE_SIZE = 10000000



def compute_sentence_complexities(dataset_path:str, complexity_function: Callable, split_clitics:bool=True, batch_size:int=16, id_offset:int=0, num_ids:int=None,**kwargs) -> list:
    sentences = []
    sentences_batch = []
    lines_to_skip = 0 # for words with clitics
    skipped_ids = 0
    if num_ids is None:
        num_ids = SAMPLE_SIZE
    with tqdm(total=SAMPLE_SIZE) as pbar:
        for line in open(dataset_path):
            if line.startswith('# sent_id = '):
                pbar.update(1)
                sent_id = line[len('# sent_id = '):].strip()
                sentence = Sentence(sent_id)
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
                skipped_ids += 1
                if skipped_ids > id_offset:
                    sentences_batch.append(sentence)
                if len(sentences)+len(sentences_batch) >= num_ids:
                    break
                if len(sentences_batch) == batch_size:
                    complexity_function(sentences_batch, **kwargs)
                    sentences += sentences_batch
                    sentences_batch = []
        if sentences_batch:
            complexity_function(sentences_batch, **kwargs)
            sentences += sentences_batch
    return sentences



def write_sentences_to_file(sentences:list, out_path:str):
    with open(out_path, 'w') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        for sentence in sentences:
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
    'gulpease': {'function': compute_gulpease_index, 'split_clitics': False}
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--sample_idx')
    parser.add_argument('-c', '--complexity_function', choices=['sentence_length', 'perplexity', 'gulpease'])
    parser.add_argument('-m', '--model_name', default='local_models/Minerva-350M-base-v1.0')
    parser.add_argument('-b', '--batch_size', type=int, default=1024)
    parser.add_argument('-o', '--ids_offset', type=int, default=0)
    parser.add_argument('-n', '--num_sentences', type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()

    conllu_path = f'/leonardo_work/IscrC_AILP/curriculum_learning/data/dataset_samples/sample_{args.sample_idx}.conllu'
    out_path = f'/leonardo_work/IscrC_AILP/curriculum_learning/data/dataset_samples/sample_{args.sample_idx}_{args.complexity_function}_{args.ids_offset}.tsv'

    kwargs = {}
    if args.complexity_function == 'perplexity':
        model, tokenizer = instantiate_model_and_tokenizer(args.model_name)
        kwargs = {'model': model, 'tokenizer': tokenizer}

    sentences = compute_sentence_complexities(conllu_path, complexity_functions[args.complexity_function]['function'], complexity_functions[args.complexity_function]['split_clitics'], batch_size=args.batch_size, id_offset=args.ids_offset, num_ids=args.num_sentences, **kwargs)
    # sorted_sentences = [sentence for sentence in sorted(sentences, key=lambda x: x.complexity)]
    sorted_sentences = sentences
    write_sentences_to_file(sorted_sentences, out_path)


if __name__ == '__main__':
    main()



