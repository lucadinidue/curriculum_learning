from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import re
import os

sns.set_style('darkgrid')

CHECKPOINTS = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000, 36000, 40000, 
               44000, 48000, 52000, 56000, 60000, 64000, 68000, 72000, 76000, 80000, 84000, 88000, 92000, 96000, 100000, 104000, 108000, 
               112000]

def map_model_name(model_name):
    return '_'.join(model_name.split('_')[5:])


def load_perplexity_df(perplexity_dir, model_type):
    model_perplexities = {}

    for model_name in os.listdir(perplexity_dir):
        file_name = f'checkpoint-{CHECKPOINTS[-1]}.csv'
        perplexity_path = os.path.join(perplexity_dir, model_name, file_name)
        perplexity_df = pd.read_csv(perplexity_path)
        model_perplexities[model_name[len(f'{model_type}_medium_'):]] = perplexity_df['perplexity'].values

    df = pd.DataFrame(model_perplexities)
    return df

def map_random_curriculum_name(orig_name, random_map):
    if 'rand' in orig_name or 'orig' in orig_name:
        splitted_name = orig_name.split('_')
        splitted_name[3] = random_map[splitted_name[3]]
        return '_'.join(splitted_name)
    return orig_name

def extract_curriculum(col):
    return re.sub(r"^\d+_train_1_", "", col)

def get_curriculum_list(df):
    groups = defaultdict(list)
    for col in df.columns:
        curr = extract_curriculum(col)
        groups[curr].append(col)
    curricula = list(groups.keys())
    return curricula, groups

def compute_perplexity_correlations(df, curricula, groups):
    corr_seed = df.corr(method="spearman")
    correlations_df = pd.DataFrame(index=curricula, columns=curricula, dtype=float)
    for currA in curricula:
        for currB in curricula:
            colsA = groups[currA]
            colsB = groups[currB]

            sub_corr = corr_seed.loc[colsA, colsB].values
            correlations_df.loc[currA, currB] = np.nanmean(sub_corr)
    
    correlations_df = correlations_df.sort_index(axis=0).sort_index(axis=1)
    return correlations_df

def plot_correlations(correlations_df, output_path):
    plt.figure(figsize=(9, 7))
    sns.heatmap(correlations_df, annot=True, fmt=".2f", cmap='coolwarm', vmin=0, vmax=1, cbar=False)
    plt.tight_layout()
    plt.show()
    plt.savefig(output_path) 



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['wikipedia', 'treebank'])
    parser.add_argument('-m', '--model_type', type=str, choices=['bert', 'gpt'])
    args = parser.parse_args()

    perplexity_dir = f'data/perplexity/{args.model_type}/{args.dataset}'
    output_path = f'results/{args.model_type}/{args.dataset}_pairwise_correlation_heatmap_last_checkpoint.png'
    df = load_perplexity_df(perplexity_dir, args.model_type)

    # Map random seeds to incremental numbers for plotting
    sorted_random_seeds = sorted(list(set([col_name.split('_')[3] for col_name in df.columns.tolist() if ('rand' in col_name or 'orig' in col_name)])))
    random_map = {orig_name: f'random_{idx+1}' for idx, orig_name  in enumerate(sorted_random_seeds)}
    df = df.rename(columns=lambda name: map_random_curriculum_name(name, random_map))

    curricula, groups = get_curriculum_list(df)
    correlations_df = compute_perplexity_correlations(df, curricula, groups)
    plot_correlations(correlations_df, output_path)
   

if __name__ == '__main__':
    main()