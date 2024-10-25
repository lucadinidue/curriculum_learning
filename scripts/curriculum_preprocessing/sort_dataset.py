from utils import load_dataset_from_csv
import pandas as pd
import argparse


def sort_dataset(df: pd.DataFrame, ascending:bool) -> pd.DataFrame:
    df = df.sort_values(by='complexity', ascending=ascending)
    return df

def sort_values(df: pd.DataFrame, vmin:int, vmax:int) -> pd.DataFrame:
    df = df[df['complexity'] >= vmin]
    df = df[df['complexity'] <= vmax]
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str)
    parser.add_argument('-o', '--output_path', type=str)
    parser.add_argument('-f', '--filter_values', action='store_true')
    parser.add_argument('-l', '--lower_value', type=float)
    parser.add_argument('-u', '--upper_value', type=float)
    parser.add_argument('-r', '--reverse_sorting', action='store_true')
    args = parser.parse_args()

    df = load_dataset_from_csv(args.input_path)
    if args.filter_values:
        if args.lower_value is None or args.upper_value is None:
            raise Exception('If "filter_values" is set to True, you must provide "lower_value" and "upper_value"')
        df = sort_values(df, vmin=args.lower_value, vmax=args.upper_value)
    ascending = not args.reverse_sorting
    sorted_df = sort_dataset(df, ascending)
    sorted_df.to_csv(args.output_path, index=False)   


if __name__ == '__main__':
    main()