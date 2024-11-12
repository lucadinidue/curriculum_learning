from utils import write_sentences_to_file, load_dataset_from_csv
from classes import Sentence
from tqdm import tqdm
import argparse
import random
import os

SAMPLE_SIZE = 10000000

def extract_sentences(dataset_path:str, out_path:str):
    sentences = []
    
    with tqdm(total=SAMPLE_SIZE) as pbar:
        for line in open(dataset_path):
            if line.startswith('# sent_id = '):
                pbar.update(1)
                sent_id = line[len('# sent_id = '):].strip()
                sentence = Sentence(sent_id)
                sentence.complexity = None
            elif line.startswith('# text = '):
                text = line[len('# text = '):].strip()
                sentence.set_text(text)
            elif line.strip() == '':
                sentences.append(sentence)
                if len(sentences) >= 1000:
                    write_sentences_to_file(sentences, out_path)
                    sentences = []
    if len(sentences) > 0:    
        write_sentences_to_file(sentences, out_path)
    return sentences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--sample_idx', type=int)
    parser.add_argument('-s', '--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is None:
      args.seed = random.randint(0, 1000)

    conllu_path = f'data/dataset_samples/sample_{args.sample_idx}.conllu'
    sentences_path = f'data/dataset_samples/sample_{args.sample_idx}_sentences.csv'
    output_path = f'data/datasets/train_{args.sample_idx}_random_{args.seed}.csv'

    if not os.path.exists(sentences_path):
        extract_sentences(conllu_path, sentences_path)

    df = load_dataset_from_csv(sentences_path)
    df = df.sample(frac=1, random_state=args.seed)
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()