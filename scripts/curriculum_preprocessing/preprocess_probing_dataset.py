import pandas as pd
import argparse
import os

def load_features_filter(src_path):
    features_filter = []
    for line in open(src_path, 'r'):
        features_filter.append(line.strip())
    return features_filter

def load_sentences_df(src_path):
    sentences_df = {'identifier':[], 'text':[]}
    for line in open(src_path):
        if line.startswith('# sent_id = '):
            sentences_df['identifier'].append(line[len('# sent_id = '):].strip())
        elif line.startswith('# text = '):
            sentences_df['text'].append(line[len('# text = '):].strip())
    df = pd.DataFrame.from_dict(sentences_df)
    df = df.set_index('identifier')
    return df



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_size', type=int, default=8000)
    parser.add_argument('-e', '--test_size', type=int, default=2000)
    parser.add_argument('-s', '--seed', type=int, default=42)
    args = parser.parse_args()

    sentences_path = 'data/probing_data/it_isdt-ud.conllu'
    features_path = 'data/probing_data/it_isdt-ud_cleaned.conllu_sent.out'
    features_filter_pathh = 'data/probing_data/filtered_features.txt'
    output_dir = 'data/probing_data'

    features_filter = load_features_filter(features_filter_pathh)

    #sentences_df = pd.read_csv(sentences_path, sep='\t', names=['identifier', 'text'], index_col='identifier')
    sentences_df = load_sentences_df(sentences_path)
    features_df = pd.read_csv(features_path, sep='\t', index_col='identifier')
    features_df = sentences_df.join(features_df)
    features_df = features_df[['text']+features_filter]

    train_df = features_df.sample(n=args.train_size, random_state=args.seed)
    features_df = features_df.drop(train_df.index)
    test_df = features_df.sample(n=args.test_size, random_state=args.seed)

    train_df.to_csv(os.path.join(output_dir, 'train.tsv'), sep='\t')
    test_df.to_csv(os.path.join(output_dir, 'test.tsv'), sep='\t')

if __name__ == '__main__':
    main()