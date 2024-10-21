from tqdm import tqdm
import argparse
import csv

def extract_sentences_sample(src_path, num_sentences):
    sentences = []
    with tqdm(total=num_sentences) as pbar:
        for line in open(src_path):
            if line.startswith('# sent_id = '):
                pbar.update(1)
                sent_id = line[len('# sent_id = '):].strip()
                sentence = [sent_id, None]
            if line.startswith('# text = '):
                sentence[1] = line[len('# text = '):].strip()
                sentences.append(sentence)
            if len(sentences) == num_sentences:
                return sentences

def write_sentences_to_file(sentences, out_path):
    with open(out_path, 'w') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        for sentence in sentences:
            csv_writer.writerow(sentence)
            
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--sample_idx', type=int)
    parser.add_argument('-d', '--sample_dim', type=int)
    args = parser.parse_args()

    conllu_path = f'data/dataset_samples/sample_{args.sample_idx}.conllu'
    out_path = f'data/dataset_samples/sample_{args.sample_idx}_eval.tsv'

    eval_sentences = extract_sentences_sample(conllu_path, args.sample_dim)
    write_sentences_to_file(eval_sentences, out_path)

if __name__ == '__main__':
    main()

    
    
