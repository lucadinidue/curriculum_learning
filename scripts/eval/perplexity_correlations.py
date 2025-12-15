from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import re
import os

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 20
})

CHECKPOINTS = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000, 36000, 40000, 
               44000, 48000, 52000, 56000, 60000, 64000, 68000, 72000, 76000, 80000, 84000, 88000, 92000, 96000, 100000, 104000, 108000, 
               112000]

NAMES_MAP = {
    'sentence_length': 'SentLen',
    'readit_global': 'ReadIt',
    'gulpease': 'Gulpease',
}


def map_model_name(model_name):
    return '_'.join(model_name.split('_')[5:])

def find_last_model_checkpoint(model_dir):
    checkpoint_dirs = [d for d in os.listdir(model_dir) if d.startswith('checkpoint-')]
    checkpoint_numbers = [int(d.split('-')[1][:-len('.csv')]) for d in checkpoint_dirs]
    last_checkpoint = max(checkpoint_numbers)
    return last_checkpoint

def load_perplexity_dfs(perplexity_dir):
    model_perplexities = {seed: {} for seed in [42, 755, 995]}

    for model_name in os.listdir(perplexity_dir):
        model_seed = int(model_name.split('_')[2])
        curriculum = '_'.join(model_name.split('_')[5:])
        model_dir = os.path.join(perplexity_dir, model_name)
        last_checkpoint = find_last_model_checkpoint(model_dir)
        file_name = f'checkpoint-{last_checkpoint}.csv'
        # file_name = f'checkpoint-{CHECKPOINTS[-1]}.csv'
        perplexity_path = os.path.join(perplexity_dir, model_name, file_name)
        perplexity_df = pd.read_csv(perplexity_path, index_col=0)
        model_perplexities[model_seed][curriculum] = perplexity_df['perplexity'].values

    for model_seed in model_perplexities.keys():
        model_perplexities[model_seed] = pd.DataFrame(model_perplexities[model_seed])

    return model_perplexities

def map_random_curriculum_name(orig_name, random_map):
    if 'rand' in orig_name or 'orig' in orig_name:
        splitted_name = orig_name.split('_')
        splitted_name[0] = random_map[splitted_name[0]]
        mapped_name =  '_'.join(splitted_name)
    else:
        if orig_name.endswith('_inverted'):
            curr_name = orig_name[:-len('_inverted')]
            mapped_name = NAMES_MAP[curr_name] + '_inverted'
        else:
            mapped_name = NAMES_MAP[orig_name]
    if mapped_name.endswith('_inverted'):
        mapped_name = mapped_name.replace('inverted', 'Inv')
    return mapped_name


def extract_curriculum(col):
    return re.sub(r"^\d+_train_1_", "", col)



def compute_perplexity_correlations(dfs):
    corr_dfs = []
    for seed in dfs.keys():
        corr_df = dfs[seed].corr(method='spearman')
        corr_dfs.append(corr_df)

    # sorted_curricula = sorted(set().union(*[c.index for c in corr_dfs]))
    sorted_curricula = ['SentLen', 'SentLen_Inv', 'Gulpease', 'Gulpease_Inv', 'ReadIt', 'ReadIt_Inv','Rand1', 'Rand1_Inv', 'Rand2', 'Rand2_Inv', 'Rand3', 'Rand3_Inv', 'Rand4', 'Rand4_Inv', 'Rand5', 'Rand5_Inv']
    aligned_correlation_matrices = [
        c.reindex(index=sorted_curricula, columns=sorted_curricula)
        for c in corr_dfs
    ]

    avg_corr = (
        pd.concat(aligned_correlation_matrices)
          .groupby(level=0, sort=False)
          .mean()
    )

    return avg_corr


def plot_correlations(correlations_df, output_path):
    plt.figure(figsize=(20, 11))
    mask = np.triu(np.ones_like(correlations_df, dtype=bool), k=1)
    ax = sns.heatmap(correlations_df, annot=True, fmt=".2f", cmap='crest', vmin=0, vmax=1, cbar=False, mask=mask)

    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')

    ax.tick_params(axis='x', pad=5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()



# def plot_correlations(correlations_df, output_path):
#     plt.figure(figsize=(15, 12))
#     sns.heatmap(correlations_df, annot=True, fmt=".2f", cmap='crest', vmin=0, vmax=1, cbar=False)
#     plt.tight_layout()
#     plt.show()
#     plt.savefig(output_path) 

# Map random seeds to incremental numbers for plotting
def map_random_seeds_to_ids(dfs):
    sorted_random_seeds = sorted(list(set([col_name.split('_')[0] for col_name in dfs[list(dfs.keys())[0]].columns.tolist() if ('rand' in col_name or 'orig' in col_name)])))
    random_map = {orig_name: f'Rand{idx+1}' for idx, orig_name  in enumerate(sorted_random_seeds)}
    for seed in dfs.keys():
        dfs[seed] = dfs[seed].rename(columns=lambda name: map_random_curriculum_name(name, random_map))
    return dfs



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['wikipedia', 'treebank'])
    parser.add_argument('-m', '--model_type', type=str, choices=['bert', 'gpt'])
    args = parser.parse_args()

    perplexity_dir = f'data/perplexity/{args.model_type}/{args.dataset}'
    output_path = f'plots/{args.model_type}/{args.dataset}_perplexity_correlations_last_epoch.pdf'
    dfs = load_perplexity_dfs(perplexity_dir)
    dfs = map_random_seeds_to_ids(dfs)

    correlations_df = compute_perplexity_correlations(dfs)
    plot_correlations(correlations_df, output_path)
   

if __name__ == '__main__':
    main()