from utils import get_seaborn_palette
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import json
import os

sns.set_style('darkgrid')

def update_loss_dict(model_name, trainer_state, loss_dict, average_random=False):
    splitted_name = model_name.split('_')
    seed = splitted_name[2]
    curriculum = '_'.join(splitted_name[splitted_name.index('1')+1:])
    if average_random:
        if 'orig' in curriculum or 'rand' in curriculum:
            curriculum = 'random'
    if loss_dict is None:
        loss_dict = {'seed':[], 'curriculum': [], 'epoch':[], 'loss': []}
    for el in trainer_state:
        if 'loss' in el:
            loss_dict['seed'].append(seed)
            loss_dict['curriculum'].append(curriculum)
            loss_dict['epoch'].append(el['epoch'])
            loss_dict['loss'].append(el['loss'])
    return loss_dict  

def load_trainer_state(model_dir):
    trainer_state_path = os.path.join(model_dir, 'trainer_state.json')
    if not os.path.exists(trainer_state_path):
        return {}
    with open(trainer_state_path, 'r') as src_file:
        trainer_state = json.load(src_file)['log_history']
    return trainer_state

def create_loss_df(models_dir, seed, average_random=False):
    loss_dict = None
    for model_name in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, model_name)
        trainer_state = load_trainer_state(model_dir)
        loss_dict = update_loss_dict(model_name, trainer_state, loss_dict, average_random)
    loss_df = pd.DataFrame.from_dict(loss_dict)
    if seed is not None:
        loss_df = loss_df[loss_df['seed'] == seed]
    # loss_df = loss_df[['curriculum', 'epoch', 'loss']]
    return loss_df

def plot_loss(loss_df, output_path):
    sorted_models = sorted(list(loss_df['curriculum'].unique()), reverse=True)
    palette = get_seaborn_palette(len(sorted_models))
    sns.lineplot(loss_df, x='epoch', y='loss', hue='curriculum', palette=palette, hue_order=sorted_models, legend=True);
    plt.savefig(output_path) 
    plt.show()

def compute_seed_loss_df(seed, models_dir, plot_output=True, output_path=None, average_random=False):
    models_dir += f'/seed_{seed}'
    loss_df = create_loss_df(models_dir, seed, average_random)
    if plot_output:
        plot_loss(loss_df, output_path)
    return loss_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_type', type=str, choices=['bert', 'gpt'])
    parser.add_argument('-s', '--model_seed', default=None, type=str)
    parser.add_argument('-a', '--average_random', action='store_true')
    args = parser.parse_args()
    
    models_dir = f'models/pretrained/{args.model_type}'
    if args.model_seed is None:
        seeds = [seed_dir.split('_')[1] for seed_dir in os.listdir(models_dir)]
    else:
        seeds = [args.model_seed]

    loss_dfs = []
    for seed in seeds:
        output_path = f'results/{args.model_type}/seed_{seed}/{args.model_type}_medium_training_loss.png'
        loss_df = compute_seed_loss_df(seed, models_dir, plot_output=args.model_seed is not None, output_path=output_path, average_random=args.average_random)    
        loss_dfs.append(loss_df)
    if args.model_seed is None:
        all_loss_dfs = pd.concat(loss_dfs)
        output_path = f'results/{args.model_type}/{args.model_type}_medium_training_loss.png'
        plot_loss(all_loss_dfs, output_path)    
        
        
if __name__ == '__main__':
    main()