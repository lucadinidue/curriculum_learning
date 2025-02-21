import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_seed', type=int)
    parser.add_argument('-o', '--output_seed', type=int)
    parser.add_argument('-t', '--task', type=str)
    args = parser.parse_args()

    src_dir = f'random_{args.source_seed}/{args.task}'
    out_dir = f'random_{args.output_seed}/{args.task}'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for file_name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, file_name)
        out_path = os.path.join(out_dir, file_name)
        with open(out_path, 'w') as out_file:
            for line in open(src_path, 'r'):
                if f'bert_medium_{args.source_seed}' in line:
                    line = line.replace(f'bert_medium_{args.source_seed}', f'bert_medium_{args.output_seed}')
                out_file.write(line)


if __name__ == '__main__':
    main()
