import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str)
    args = parser.parse_args()

    src_dir = args.task
    out_dir = src_dir+'_2'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for file_name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, file_name)
        out_path = os.path.join(out_dir, file_name)
        
        with open(out_path, 'w') as out_file:
            for line in open(src_path, 'r'):
                if '--time' in line:
                    line = '#SBATCH --time 02:00:00     # format: HH:MM:SS\n'
                out_file.write(line)

if __name__ == '__main__':
    main()
