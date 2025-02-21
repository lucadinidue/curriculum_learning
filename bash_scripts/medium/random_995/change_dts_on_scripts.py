import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_task', type=str)
    parser.add_argument('-o', '--output_task', type=str)
    args = parser.parse_args()

    src_dir = args.source_task
    out_dir = args.output_task

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for file_name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, file_name)
        out_path = os.path.join(out_dir, file_name)

        with open(out_path, 'w') as out_file:
            for line in open(src_path, 'r'):
                if args.source_task in line:
                    line = line.replace(args.source_task, args.output_task)
                out_file.write(line)

if __name__ == '__main__':
    main()
