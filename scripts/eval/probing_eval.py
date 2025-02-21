from utils import get_seaborn_palette
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import csv
import os

sns.set_style("darkgrid")

HUE_ORDER = ['sentence_length_inverted', 'sentence_length', 'readit_global_inverted', 'readit_global', 'gulpease_inverted', 'gulpease', 'random']

def score_model(src_path):
    predictions, labels = [], []
    with open(src_path, 'r') as src_file:
        csv_reader = csv.reader(src_file, delimiter='\t')
        for row in csv_reader:
            predictions.append(float(row[0]))
            labels.append(float(row[1]))
        s = spearmanr(predictions, labels)
    return s.statistic if s.pvalue < 0.05 else None

def add_to_res_dict(res_dict, model, checkpoint, feature, layer, score):
    res_dict['model'].append(model)
    res_dict['checkpoint'].append(checkpoint)
    res_dict['feature'].append(feature)
    res_dict['layer'].append(layer)
    res_dict['score'].append(score)

def normalize_model_name(model_name, model_seed, average_random):
    if model_seed is None:
        model_name = '_'.join(model_name.split('_')[5:])
    if average_random:
        if 'rand' in model_name or 'orig' in model_name:
            model_name = 'random'
    return model_name

def load_res_df(src_dir, res_df, model_seed=None, average_random=False):
    res_dict = {'model': [], 'checkpoint':[], 'feature': [], 'layer':[], 'score':[]}

    for model_name in os.listdir(src_dir):
        if model_seed and f'{model_seed}_train' not in model_name:
            continue
        print(model_name)
        model_dir = os.path.join(src_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        for checkpoint_dir_name in sorted(os.listdir(model_dir)):
            checkpoint_num = int(checkpoint_dir_name.split('-')[-1])
            already_computed = ((res_df['model'] == model_name) & (res_df['checkpoint'] == checkpoint_num)).any()
            if not already_computed:
                checkpoint_dir = os.path.join(model_dir, checkpoint_dir_name)
                try:
                    for layer in os.listdir(checkpoint_dir):
                        layer_dir = os.path.join(checkpoint_dir, layer)
                        for feature_file_name in os.listdir(layer_dir):
                            feature = feature_file_name[:-4]
                            feature_file_path = os.path.join(layer_dir, feature_file_name)
                            score = score_model(feature_file_path)
                            add_to_res_dict(res_dict, model_name, checkpoint_num, feature, int(layer)+1, score)
                except Exception as e:
                    print(e)
                    print(checkpoint_dir)
    
    return pd.concat([res_df, pd.DataFrame.from_dict(res_dict)])

def print_features_scores(res_df, output_path,  average_random=False, legend_path=None, max_checkpoint=None):
    sorted_models = HUE_ORDER if average_random else sorted(list(res_df['model'].unique()), reverse=True)
    print(sorted_models)
    palette = get_seaborn_palette(len(sorted_models))
    if max_checkpoint is not None:
        res_df = res_df[res_df['checkpoint'] <= max_checkpoint]
        
    features = sorted(list(res_df['feature'].unique()))
    layers = sorted(list(res_df['layer'].unique()))
   
    n_rows = len(features)
    n_cols = len(layers)
    _, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4 * n_rows))

    for feat_idx, feature in enumerate(features):
        for layer_idx, layer in enumerate(layers):
            vmin = res_df[res_df['feature'] == feature]['score'].min()
            vmax = res_df[res_df['feature'] == feature]['score'].max()
            feature_df = res_df[(res_df['feature'] == feature) & (res_df['layer'] == layer)]
            sns.lineplot(data=feature_df, x='checkpoint', y='score', hue='model', marker='o', hue_order=sorted_models, palette=palette, ax=axes[feat_idx][layer_idx], legend=False)
            axes[feat_idx][layer_idx].set_title(f'{feature}, layer {layer}')
            for layer_idx in range(len(layers)):
                axes[feat_idx][layer_idx].set_ylim(vmin-0.05, vmax+0.05)
    plt.tight_layout(h_pad=2)
    plt.savefig(output_path)
    plt.show()

    if legend_path is not None:
        # legend
        legend_plot = sns.lineplot(data=feature_df, x='checkpoint', y='score', hue='model', marker='o', hue_order=sorted_models, palette=palette)
        handles, labels = legend_plot.get_legend_handles_labels()
        plt.close()
        plt.figure()
        plt.legend(handles=handles, labels=labels)#, loc='center', frameon=False)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(legend_path)
        plt.show()


def load_computed_correlations(src_path):
    if os.path.exists(src_path):
        print('Loaded correlations.')
        return pd.read_csv(src_path, index_col=0)
    else:
        return pd.DataFrame(columns=['model', 'checkpoint', 'feature', 'layer', 'score'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', type=str, default='data/probing_results/medium')
    parser.add_argument('-s', '--model_seed', type=int, choices=[42, 755, 995, None])
    parser.add_argument('-a', '--average_random', action='store_true')
    parser.add_argument('-c', '--correlations_path', type=str, default=None)
    parser.add_argument('-o', '--output_path', type=str)
    parser.add_argument('-l', '--legend_path', type=str, default=None)
    args = parser.parse_args()

    already_computed_df = load_computed_correlations(args.correlations_path)
    res_df = load_res_df(args.input_directory, already_computed_df,  model_seed=args.model_seed, average_random=args.average_random)
    res_df.to_csv(args.correlations_path)
    res_df['model'] = res_df.apply(lambda row: normalize_model_name(row['model'], model_seed=args.model_seed, average_random=args.average_random), axis=1)
    print('Computing plot')
    print_features_scores(res_df, args.output_path, average_random=args.average_random, legend_path= args.legend_path, max_checkpoint=None)

if __name__ == '__main__':
    main()