from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

sns.set_style("darkgrid")

HUE_ORDER = ['sentence_length_inverted', 'sentence_length', 'readit_global_inverted', 'readit_global', 'gulpease_inverted', 'gulpease']
CHECKPOINTS = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000, 36000, 40000, 
               44000, 48000, 52000, 56000, 60000, 64000, 68000, 72000, 76000, 80000, 84000, 88000, 92000, 96000, 100000, 104000, 108000, 
               112000]


def load_correlations_df(correlations_path, min_chekpoint=None, last_checkpoint=False, last_layer=False):
    df = pd.read_csv(correlations_path, index_col=0)
    df = df[df['checkpoint'].isin(CHECKPOINTS)]
    if last_checkpoint:
        max_checkpoint = df['checkpoint'].max()
        df = df[df['checkpoint'] == max_checkpoint]
    if last_layer:
        df = df[df['layer'] == 8]
    if min_chekpoint is not None:
        df = df[df['checkpoint'] >= min_chekpoint]
    return df

def extract_curriculum(row_model):
    curriculum = '_'.join(row_model.split('_')[5:])
    if 'rand' in curriculum or 'orig' in curriculum:
        curriculum = 'random'
    return curriculum


def compute_z_scores_df(df):
    z_dfs = []

    for feature in df['feature'].unique():
        for layer in df['layer'].unique():
            filtered_df = df[(df['feature'] == feature) & (df['layer'] == layer)]
            random_scores = filtered_df[filtered_df['curriculum'] == 'random']['score']
            random_mean, random_std = random_scores.mean(), random_scores.std()
            if random_std == 0:
                continue
            filtered_df = filtered_df.copy()
            filtered_df['z'] = (filtered_df['score'] - random_mean) / random_std
            z_dfs.append(filtered_df)

    return pd.concat(z_dfs, ignore_index=True)



def compute_statistical_significance(z_df):
    results = []
    for feature in z_df['feature'].unique():
        for curr in z_df['curriculum'].unique():
            if curr == 'random':
                continue
            rand_vals = z_df[(z_df['feature'] == feature) & (z_df['curriculum'] == 'random')].sort_values('checkpoint')['z']
            curr_vals = z_df[(z_df['feature'] == feature) & (z_df['curriculum'] == curr)].sort_values('checkpoint')['z']
        
            t, p = ttest_ind(curr_vals, rand_vals, equal_var=False)
            diff = curr_vals.mean() - rand_vals.mean()
            results.append({
                'feature': feature,
                'curriculum': curr,
                'mean_z_diff': diff,
                'pval': p
            })

    results_df = pd.DataFrame(results)
    reject, p_corr, _, _ = multipletests(results_df['pval'], method='fdr_bh')
    results_df['pval_corrected'] = p_corr
    results_df['significant'] = reject

    return results_df


def get_top_significant_features(df):
    sorted_features = (
        df[df['significant'] & (df['mean_z_diff'] > 0)]
        .sort_values('mean_z_diff', ascending=False)
    )

    top_features = []
    for feature in sorted_features['feature']:
        if feature not in top_features:
            top_features.append(feature)

    return top_features

def plot_feature_z_scores(df, top_features, output_path, n=10):
    top_df = df[df['feature'].isin(top_features[:n])]

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=top_df,
        x='feature',
        y='z',
        hue='curriculum',
        hue_order=HUE_ORDER,
        palette=sns.color_palette("Paired", len(HUE_ORDER)),
        order=top_features[:n]
        )
    plt.xticks(rotation=45);

    plt.savefig(output_path, bbox_inches='tight')


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_type', '-m', type=str, choices=['bert', 'gpt'])
    argparser.add_argument('--start_checkpoint', '-s', type=int, default=None)
    argparser.add_argument('--output_path', '-o', type=str, default=None)
    argparser.add_argument('--only_last_checkpoint', '-c', action='store_true')
    argparser.add_argument('--only_last_layer', '-l', action='store_true')
    args = argparser.parse_args()

    correlations_path = f'data/probing_results/{args.model_type}.json'
    #output_path = f'results/{args.model_type}/probing_top_significant_higher.png'
    df = load_correlations_df(correlations_path, min_chekpoint=args.start_checkpoint, last_checkpoint=args.only_last_checkpoint, last_layer=args.only_last_layer)
    
    # df['seed'] = df['model'].str.extract(r'_(\d+)_')[0].astype(int)
    df['curriculum'] = df['model'].apply(extract_curriculum)

    z_df = compute_z_scores_df(df)
    result_df = compute_statistical_significance(z_df)
    top_features = get_top_significant_features(result_df)

    plot_feature_z_scores(z_df, top_features, args.output_path, n=10)

if __name__ == '__main__':
    main()