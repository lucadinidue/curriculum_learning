from utils import load_dataset_from_csv
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str)
    args = parser.parse_args()

    output_path = args.input_file[:-len('.csv')]+'_inverted.csv'

    df = load_dataset_from_csv(args.input_file)
    df = df.iloc[::-1]
    df.to_csv(output_path, index=False)   

if __name__ == '__main__':
    main()