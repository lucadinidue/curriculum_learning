from tqdm import tqdm
import argparse
import csv
import os

SAMPLE_SIZE = 10000000


def compute_sentence_length(tokens):
    last_token_id = tokens[-1].split('\t')[0]
    return int(last_token_id)

    
def compute_sentence_complexities(dataset_path, complexity_function, split_clitics=True):
    sentences = []
    with tqdm(total=SAMPLE_SIZE) as pbar:
        for line in open(dataset_path):
            if line.startswith('# sent_id = '):
                pbar.update(1)
                tokens = []
                sent_id = line[len('# sent_id = '):].strip()
                sentence = [sent_id, None, None]
            if line.startswith('# text = '):
                sentence[1] = line[len('# text = '):].strip()
                
            if line[0].isdigit():
                token_id = line.split('\t')[0]
                if split_clitics:
                    if '-' not in token_id:
                        tokens.append(line.strip())
                else:
                    raise Exception('Not implemented yet')
            
            if line.strip() == '':
                try:
                    complexity = complexity_function(tokens)
                except:
                    print(sentence)
                sentence[2] = complexity
                sentences.append(sentence)
    return sentences


def write_sentences_to_file(sentences, out_path):
    with open(out_path, 'w') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        for sentence in sentences:
            csv_writer.writerow(sentence)

complexity_fuctions = {
    'sentence_length': compute_sentence_length
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--sample_idx')
    parser.add_argument('-c', '--complexity_function', choices=['sentence_length'])
    args = parser.parse_args()

    conllu_path = f'data/dataset_samples/sample_{args.sample_idx}.conllu'
    out_path = f'data/dataset_samples/sample_{args.sample_idx}_{args.complexity_function}.tsv'
    
    sentences = compute_sentence_complexities(conllu_path, complexity_fuctions[args.complexity_function])
    sorted_sentences = [sentence for sentence in sorted(sentences, key=lambda x: x[2])]
    write_sentences_to_file(sorted_sentences, out_path)


if __name__ == '__main__':
    main()



