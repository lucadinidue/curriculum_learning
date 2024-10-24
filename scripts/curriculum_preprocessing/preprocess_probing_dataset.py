import pandas as pd
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_size', type=int, default=8000)
    parser.add_argument('-e', '--test_size', type=int, default=2000)
    parser.add_argument('-s', '--seed', type=int, default=42)
    args = parser.parse_args()

    sentences_path = '/home/wiki_pretraining/probing_data/Italian_UD.tsv'
    features_path = '/home/wiki_pretraining/probing_data/Italian_UD.conllu_sent.out'

    output_dir = 'data/probing_data'

    sentences_df = pd.read_csv(sentences_path, sep='\t', names=['identifier', 'text'], index_col='identifier')
    features_df = pd.read_csv(features_path, sep='\t', index_col='identifier')
    features_df = sentences_df.join(features_df)

    train_df = features_df.sample(n=args.train_size, random_state=args.seed)
    features_df = features_df.drop(train_df.index)
    test_df = features_df.sample(n=args.test_size, random_state=args.seed)

    train_df.to_csv(os.path.join(output_dir, 'train.tsv'), sep='\t')
    test_df.to_csv(os.path.join(output_dir, 'test.tsv'), sep='\t')

if __name__ == '__main__':
    main()