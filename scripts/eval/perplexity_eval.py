from utils import get_seaborn_palette
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os

sns.set_style("darkgrid")

def score_model(src_path):
    scores_df = pd.read_csv(src_path)
    return scores_df['plausibility'].tolist(), scores_df['perplexity'].tolist()
    # avg_plausibility = scores_df['plausibility'].mean()
    # avg_perplexity = scores_df['perplexity'].mean()
    # return avg_plausibility, avg_perplexity

def add_to_res_dict(res_dict, model, checkpoint, metric, score, dataset):
    res_dict['model'].append(model)
    res_dict['checkpoint'].append(checkpoint)
    res_dict['metric'].append(metric)
    res_dict['score'].append(score)
    res_dict['dataset'].append(dataset)


def load_perplexity_df(src_dir, model_size, model_seed, average_random=False):
    res_dict = {'model': [], 'checkpoint':[], 'metric': [], 'score':[], 'dataset':[]}
    for dataset in ['wikipedia', 'treebank']:
        dataset_dir = os.path.join(src_dir, dataset)
        for model_name in os.listdir(dataset_dir):
            if model_size not in model_name or f'{model_seed}_train' not in model_name:
                continue
            model_dir = os.path.join(dataset_dir, model_name)
            for checkpoint_file_name in os.listdir(model_dir):
                
                checkpoint_num = int(checkpoint_file_name.split('-')[-1][:-4])
                checkpoint_path = os.path.join(model_dir, checkpoint_file_name)

                plausibility, perplexity = score_model(checkpoint_path)
                if '_rand' in model_name or 'orig' in model_name:
                    if average_random:
                        model_name = 'random'
                for pl, pp in zip(plausibility, perplexity):
                    add_to_res_dict(res_dict, model_name, checkpoint_num, 'plausibility', pl, dataset)
                    add_to_res_dict(res_dict, model_name, checkpoint_num, 'perplexity', pp, dataset)        
    return pd.DataFrame.from_dict(res_dict)

def plot_results(res_df, output_path, min_checkpoint=None):
    ordered_models = sorted(res_df['model'].unique().tolist(), reverse=True)
    palette = get_seaborn_palette(len(ordered_models))
    for metric in sorted(list(res_df['metric'].unique())):
        metric_df = res_df[res_df['metric'] == metric]
        if min_checkpoint is not None:
            metric_df = metric_df[metric_df['checkpoint'] >= min_checkpoint]
        _, axes = plt.subplots(2, 1, figsize=(10, 15))
        for idx, dataset in enumerate(['treebank', 'wikipedia']):
            dataset_df = metric_df[metric_df['dataset'] == dataset]
            sns.lineplot(data=dataset_df, x='checkpoint', y='score', hue='model', hue_order=ordered_models, palette=palette, marker='o', legend=True, ax=axes[idx])
            axes[idx].set_title(f'{metric}, {dataset}')
        plt.savefig(f'{output_path}_{metric}.png') 
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--model_size', default='medium', choices=['small', 'medium', 'base'])
    parser.add_argument('-s', '--model_seed', type=int, choices=[42, 755, 995])
    parser.add_argument('-a', '--average_random', action='store_true')
    args = parser.parse_args()

    src_dir = 'data/perplexity/'
    output_path = f'results/seed_{args.model_seed}/bert_{args.model_size}'
    if args.average_random:
        output_path += '_avg_random'
    res_df = load_perplexity_df(src_dir, args.model_size, args.model_seed, average_random=args.average_random)
    plot_results(res_df, output_path)

if __name__ == '__main__':
    main()

