from utils import get_seaborn_palette
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os

sns.set_style("darkgrid")

HUE_ORDER = ['sentence_length_inverted', 'sentence_length', 'readit_global_inverted', 'readit_global', 'gulpease_inverted', 'gulpease', 'random']

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

def normalize_model_name(model_name, model_seed, average_random):
    if model_seed is None:
        model_name = '_'.join(model_name.split('_')[5:])
    if average_random:
        if 'rand' in model_name or 'orig' in model_name:
            model_name = 'random'
    return model_name

def load_perplexity_df(src_dir, model_size, model_seed, average_random=False):
    res_dict = {'model': [], 'checkpoint':[], 'metric': [], 'score':[], 'dataset':[]}
    for dataset in ['wikipedia', 'treebank']:
        dataset_dir = os.path.join(src_dir, dataset)
        for model_name in os.listdir(dataset_dir):
            if model_size not in model_name or (model_seed and f'{model_seed}_train' not in model_name):
                continue
            model_dir = os.path.join(dataset_dir, model_name)
            model_name = normalize_model_name(model_name, model_seed, average_random)
            for checkpoint_file_name in os.listdir(model_dir):
                checkpoint_num = int(checkpoint_file_name.split('-')[-1][:-4])
                checkpoint_path = os.path.join(model_dir, checkpoint_file_name)

                plausibility, perplexity = score_model(checkpoint_path)
                for pl, pp in zip(plausibility, perplexity):
                    add_to_res_dict(res_dict, model_name, checkpoint_num, 'plausibility', pl, dataset)
                    add_to_res_dict(res_dict, model_name, checkpoint_num, 'perplexity', pp, dataset)        
    return pd.DataFrame.from_dict(res_dict)

def plot_results(res_df, output_path, average_random=False, min_checkpoint=None):
    sorted_models = HUE_ORDER if average_random else sorted(list(res_df['model'].unique()), reverse=True)
    palette = get_seaborn_palette(len(sorted_models))
    for metric in sorted(list(res_df['metric'].unique())):
        metric_df = res_df[res_df['metric'] == metric]
        if min_checkpoint is not None:
            metric_df = metric_df[metric_df['checkpoint'] >= min_checkpoint]
        _, axes = plt.subplots(2, 1, figsize=(10, 15))
        for idx, dataset in enumerate(['treebank', 'wikipedia']):
            dataset_df = metric_df[metric_df['dataset'] == dataset]
            sns.lineplot(data=dataset_df, x='checkpoint', y='score', hue='model', hue_order=sorted_models, palette=palette, marker='o', legend=True, ax=axes[idx])
            axes[idx].set_title(f'{metric}, {dataset}')
        plt.savefig(f'{output_path}_{metric}.png') 
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--model_size', default='medium', choices=['small', 'medium', 'base'])
    parser.add_argument('-s', '--model_seed', type=int, choices=[42, 755, 995, None])
    parser.add_argument('-a', '--average_random', action='store_true')
    args = parser.parse_args()

    src_dir = 'data/perplexity/'
    if args.model_seed:
        output_path = f'results/seed_{args.model_seed}/bert_{args.model_size}'
    else:
        output_path = f'results/bert_{args.model_size}'
    res_df = load_perplexity_df(src_dir, args.model_size, args.model_seed, average_random=args.average_random)
    plot_results(res_df, output_path, average_random=args.average_random)

if __name__ == '__main__':
    main()

