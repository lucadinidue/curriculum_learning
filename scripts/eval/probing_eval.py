from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import csv
import os

sns.set_style("darkgrid")


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


def load_res_df(src_dir):
    res_dict = {'model': [], 'checkpoint':[], 'feature': [], 'layer':[], 'score':[]}

    for model_name in os.listdir(src_dir):
        model_dir = os.path.join(src_dir, model_name)
        for checkpoint_dir_name in sorted(os.listdir(model_dir)):
            checkpoint_num = int(checkpoint_dir_name.split('-')[-1])
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
    
    return pd.DataFrame.from_dict(res_dict)

def print_features_scores(res_df, output_path, legend_path=None, max_checkpoint=None):
    sorted_models = sorted(list(res_df['model'].unique()), reverse=True)

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
            sns.lineplot(data=feature_df, x='checkpoint', y='score', hue='model', marker='o', hue_order=sorted_models, palette='Paired', ax=axes[feat_idx][layer_idx], legend=False)
            axes[feat_idx][layer_idx].set_title(f'{feature}, layer {layer}')
            for layer_idx in range(len(layers)):
                axes[feat_idx][layer_idx].set_ylim(vmin-0.05, vmax+0.05)
    plt.tight_layout(h_pad=2)
    plt.savefig(output_path)
    plt.show()

    if legend_path is not None:
        # legend
        legend_plot = sns.lineplot(data=feature_df, x='checkpoint', y='score', hue='model', marker='o', hue_order=sorted_models, palette='Paired')
        handles, labels = legend_plot.get_legend_handles_labels()
        plt.close()
        plt.figure()
        plt.legend(handles=handles, labels=labels)#, loc='center', frameon=False)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(legend_path)
        plt.show()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', type=str, default='../../data/probing_results/pretrained')
    parser.add_argument('-o', '--ouput_path', type=str)
    parser.add_argument('-l', '--legend_path', type=str, default=None)
    args = parser.parse_args()

    res_df = load_res_df(args.input_directory)
    print_features_scores(res_df, args.output_path, args.legend_path, max_checkpoint=None)

if __name__ == '__main__':
    main()