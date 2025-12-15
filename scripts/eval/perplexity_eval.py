from utils import get_seaborn_palette, map_checkpoints_to_tokens, normalize_model_name, aggregate_random_results
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import json
import os

sns.set_style("darkgrid")


plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 28,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20
})

HUE_ORDER = ['sentence_length_inverted', 'sentence_length', 'readit_global_inverted', 'readit_global', 'gulpease_inverted', 'gulpease', 'random']

def score_model(src_path):
    scores_df = pd.read_csv(src_path)
    return scores_df['perplexity'].tolist(), None if 'plausibility' not in scores_df.columns else scores_df['plausibility'].tolist() 
    # avg_plausibility = scores_df['plausibility'].mean()
    # avg_perplexity = scores_df['perplexity'].mean()
    # return avg_plausibility, avg_perplexity

def add_to_res_dict(res_dict, model, checkpoint, metric, score, dataset):
    res_dict['model'].append(model)
    res_dict['checkpoint'].append(checkpoint)
    res_dict['metric'].append(metric)
    res_dict['score'].append(score)
    res_dict['dataset'].append(dataset)


def load_perplexity_df(src_dir, model_seed):
    res_dict = {'model': [], 'checkpoint':[], 'metric': [], 'score':[], 'dataset':[]}
    for dataset in ['wikipedia', 'treebank']:
        dataset_dir = os.path.join(src_dir, dataset)
        for model_name in os.listdir(dataset_dir):
            if (model_seed and f'{model_seed}_train' not in model_name):
                continue
            model_dir = os.path.join(dataset_dir, model_name)
            for checkpoint_file_name in os.listdir(model_dir):
                checkpoint_num = int(checkpoint_file_name.split('-')[-1][:-4])
                checkpoint_path = os.path.join(model_dir, checkpoint_file_name)

                perplexity, plausibility = score_model(checkpoint_path)
                if plausibility is not None:
                    for pl, pp in zip(plausibility, perplexity):
                        add_to_res_dict(res_dict, model_name, checkpoint_num, 'plausibility', pl, dataset)
                        add_to_res_dict(res_dict, model_name, checkpoint_num, 'perplexity', pp, dataset) 
                else:
                    for pp in perplexity:
                        add_to_res_dict(res_dict, model_name, checkpoint_num, 'perplexity', pp, dataset)  
    return pd.DataFrame.from_dict(res_dict)



def plot_results(res_df, output_path, model_seed, average_random=False, min_checkpoint=None, checkpoint_tokens_map=None):
    sorted_models = HUE_ORDER if average_random else sorted(list(res_df['model'].unique()), reverse=True)
    palette = get_seaborn_palette(len(sorted_models))

    if checkpoint_tokens_map is not None:
        res_df = map_checkpoints_to_tokens(res_df, checkpoint_tokens_map)

    res_df['model'] = res_df['model'].apply(lambda x: normalize_model_name(x, model_seed, average_random))
    x_key = 'num_training_tokens' if checkpoint_tokens_map is not None else 'checkpoint'

    if checkpoint_tokens_map:
        res_df = aggregate_random_results(res_df)

    for metric in sorted(list(res_df['metric'].unique())):
        metric_df = res_df[res_df['metric'] == metric]

        _, axes = plt.subplots(2, 1, figsize=(10, 16))
        plt.subplots_adjust(hspace=0.35)  
        for idx, dataset in enumerate(['treebank', 'wikipedia']):
            dataset_df = metric_df[metric_df['dataset'] == dataset]
            sns.lineplot(data=dataset_df, x=x_key, y='score', hue='model', hue_order=sorted_models, palette=palette, marker='o', legend=False, ax=axes[idx])
            axes[idx].set_title(f'{dataset}')
            # ymax = dataset_df['score'].quantile(0.80)
            # line_mins = []
            # for line in axes[idx].lines:
            #     ydata = line.get_ydata()
            #     if len(ydata) > 0:
            #         line_mins.append(min(ydata))
            # ymin = min(line_mins)-30
            # axes[idx].set_ylim(ymin=ymin, ymax=ymax)
            # axes[idx].axhline(y=ymax, color='black', linestyle='--', alpha=0.6)
            # axes[idx].text(
            #     x=0.02, 
            #     y=ymax - 0.05*(ymax - ymin),  # push text slightly down
            #     s='values above this line not shown',
            #     transform=axes[idx].transData,
            #     fontsize=21, 
            #     color='black',    
            #     va='bottom'
            # )           

            axes[idx].set_ylabel("perplexity")
            axes[idx].set_xlabel("training tokens" if x_key == "num_training_tokens" else "checkpoint")

        plt.tight_layout()
        plt.savefig(f'{output_path}{metric}.pdf') 
        plt.show()


def load_checkpoint_tokens_map(src_dir):
    checkpoint_tokens_map = {}
    for curriculum_file_name in os.listdir(src_dir):
        curriculum = curriculum_file_name.split('.')[0]
        file_path = os.path.join(src_dir, curriculum_file_name)
        with open(file_path, 'r') as src_file:
            curriculum_map = json.load(src_file)
        checkpoint_tokens_map[curriculum] = curriculum_map
    return checkpoint_tokens_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_type', choices=['bert', 'gpt'])
    parser.add_argument('-s', '--model_seed')
    parser.add_argument('-a', '--average_random', action='store_true')
    parser.add_argument('-n', '--num_tokens_map', action='store_true')
    args = parser.parse_args()

    if args.num_tokens_map:
        num_tokens_dir = f'data/num_training_tokens/{args.model_type}'
        checkpoint_tokens_map = load_checkpoint_tokens_map(num_tokens_dir)
    else:
        checkpoint_tokens_map = None

    src_dir = f'data/perplexity/{args.model_type}'
    if args.model_seed:
        output_path = f'plots/{args.model_type}/seed_{args.model_seed}/'
    else:
        output_path = f'plots/{args.model_type}/'
    # if args.num_tokens_map:
    #     output_path = 'num_tokens_' + output_path
    res_df = load_perplexity_df(src_dir, args.model_seed)
    plot_results(res_df, output_path, model_seed=args.model_seed, average_random=args.average_random, checkpoint_tokens_map=checkpoint_tokens_map)

if __name__ == '__main__':
    main()

