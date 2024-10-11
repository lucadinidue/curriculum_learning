import pandas as pd
import argparse

def load_dataset(src_path:str) -> pd.DataFrame:
    df = pd.read_csv(src_path, names=['sent_id', 'text', 'complexity'])
    return df

def sort_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by='complexity')
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path')
    parser.add_argument('-o', '--output_path')
    args = parser.parse_args()

    df = load_dataset(args.input_path)
    sorted_df = sort_dataset(df)
    sorted_df.to_csv(args.output_path)   


if __name__ == '__main__':
    main()