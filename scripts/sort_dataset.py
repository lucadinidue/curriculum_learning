from utils import load_dataset_from_csv
import pandas as pd
import argparse


def sort_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by='complexity')
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path')
    parser.add_argument('-o', '--output_path')
    args = parser.parse_args()

    df = load_dataset_from_csv(args.input_path)
    sorted_df = sort_dataset(df)
    sorted_df.to_csv(args.output_path)   


if __name__ == '__main__':
    main()