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
    with open(trainer_state_path, 'r') as src_file:
        trainer_state = json.load(src_file)['log_history']
    return trainer_state

def create_loss_df(models_dir, seed):
    loss_dict = None
    for model_name in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, model_name)
        trainer_state = load_trainer_state(model_dir)
        loss_dict = update_loss_dict(model_name, trainer_state, loss_dict)
    loss_df = pd.DataFrame.from_dict(loss_dict)

    seed_df = loss_df[loss_df['seed'] == seed][['curriculum', 'epoch', 'loss']]
    return seed_df

def plot_loss(loss_df, output_path):
    sorted_models = sorted(list(loss_df['curriculum'].unique()), reverse=True)
    palette = get_seaborn_palette(len(sorted_models))
    sns.lineplot(loss_df, x='epoch', y='loss', hue='curriculum', palette=palette, hue_order=sorted_models, legend=False);
    plt.savefig(output_path) 
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models_dir', type=str)
    parser.add_argument('-s', '--model_seed')
    args = parser.parse_args()
    
    output_path = f'results/seed_{args.model_seed}/bert_medium_training_loss.png'
    loss_df = create_loss_df(args.models_dir, args.model_seed)
    plot_loss(loss_df, output_path)
    

if __name__ == '__main__':
    main()