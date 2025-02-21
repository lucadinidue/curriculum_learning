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
                if '--time' in line:
                    line = '#SBATCH --time 08:30:00     # format: HH:MM:SS\n'
                elif '--gres=gpu'in line:
                    line = '#SBATCH --gres=gpu:2        # 4 gpus per node out of 4\n'
                elif 'python scripts/train_mlm.py' in line:
                    config_name = line.strip().split('/')[-1]
                    line = f'python scripts/train_mlm.py /leonardo_work/IscrC_AILP/curriculum_learning/data/configs/training_configs/medium/random_{args.output_seed}/' + config_name + '\n'
                    
                if 'scripts/compute_sentence_complexity.py' in line:
                    line = ''
                out_file.write(line)


if __name__ == '__main__':
    main()
