from utils import get_seaborn_palette
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import json
import os

sns.set_style("darkgrid")

TASKS_METRICS = {
    'coherence': ['accuracy', 'f1'],
    'complexity': ['mae', 'spearmanr'],
    'sentiment': ['neg_f1', 'pos_f1'],#['neg_accuracy', 'neg_f1', 'pos_accuracy', 'pos_f1'],
    'pos_tagging': ['f1', 'accuracy']
}

HUE_ORDER = ['sentence_length_inverted', 'sentence_length', 'readit_global_inverted', 'readit_global', 'gulpease_inverted', 'gulpease', 'random']


def load_trainer_state(src_path):
    with open(src_path, 'r') as src_file:
        trainer_state = json.load(src_file)
    return trainer_state

def get_last_epoch_eval_metrics(trainer_state, metrics):
    res = {metric: None for metric in metrics}
    last_epoch = 0 
    for el in trainer_state['log_history']:
        if 'eval_loss' in el:
            for metric in metrics:
                res[metric] = el[f'eval_{metric}']
            last_epoch = int(el['epoch'])
    assert last_epoch == 10
    return res

def add_to_result_dict(res_dict, checkpoint, eval_metrics):
    for metric in eval_metrics:
        res_dict['checkpoint'].append(checkpoint)
        res_dict['metric'].append(metric)
        res_dict['score'].append(eval_metrics[metric])

def get_model_results(models_dir, model_name, downstream_task, average_metrics=False):
    res_dict = {'checkpoint':[], 'metric':[], 'score':[]}
    model_dir = os.path.join(models_dir, model_name)
    for checkpoint_name in os.listdir(model_dir):
        checkpoint_num = int(checkpoint_name.split('-')[-1])
        trainer_state_path = os.path.join(model_dir, checkpoint_name, 'trainer_state.json')
        try:
            trainer_state = load_trainer_state(trainer_state_path)
        except FileNotFoundError:
            continue
        eval_metrics = get_last_epoch_eval_metrics(trainer_state, metrics=TASKS_METRICS[downstream_task])
        if average_metrics:
            eval_metrics = {'score': sum(list(eval_metrics.values()))/len(eval_metrics)}
        add_to_result_dict(res_dict, checkpoint_num, eval_metrics)
    res_df = pd.DataFrame.from_dict(res_dict)
    return res_df

def normalize_model_name(model_name, model_seed, average_random):
    if model_seed is None:
        model_name = '_'.join(model_name.split('_')[5:])
    if average_random:
        if 'rand' in model_name or 'orig' in model_name:
            model_name = 'random'
    return model_name

def get_task_results(models_dir, downstream_task, model_size, model_seed=None, average_random=False, average_metrics=False):
    res_df = []
    for model_name in os.listdir(models_dir):
        if model_size in model_name:
            if model_seed is None or f'{model_seed}_train' in model_name:
                model_df = get_model_results(models_dir, model_name, downstream_task, average_metrics)
                model_df['model'] = normalize_model_name(model_name, model_seed, average_random)
                res_df.append(model_df)
    res_df = pd.concat(res_df)
    return res_df

def map_checkpoints_to_tokens(res_df, checkpoint_tokens_map):
    def map_row(row):
        curriculum = '_'.join(row['model'].split('_')[5:])
        checkpoint = str(row['checkpoint'])
        num_tokens = None
        if curriculum in checkpoint_tokens_map:
            if checkpoint in checkpoint_tokens_map[curriculum]:
                num_tokens = checkpoint_tokens_map[curriculum][checkpoint]
                return num_tokens
        return None
    
    res_df['num_training_tokens'] = res_df.apply(map_row, axis=1)
    res_df = res_df[res_df['num_training_tokens'].notnull()]
    return res_df


def plot_results(res_df, task, output_path, checkpoint_tokens_map=None, average_random=False, max_checkpoint=None, metric_name=None):
    # sorted_models = HUE_ORDER if average_random else 
    sorted_models = sorted(list(res_df['model'].unique()), reverse=True)
    # sorted_models[0], sorted_models[-1] = sorted_models[-1], sorted_models[0]    
    palette = get_seaborn_palette(len(sorted_models))

    if max_checkpoint is not None:
        res_df = res_df[res_df['checkpoint'] <= max_checkpoint]

    if checkpoint_tokens_map is not None:
        res_df = map_checkpoints_to_tokens(res_df, checkpoint_tokens_map)

    x_key = 'num_training_tokens' if checkpoint_tokens_map is not None else 'checkpoint'

    metrics = list(res_df['metric'].unique())
    _, axes = plt.subplots(len(metrics), 1, figsize=(10, 7*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        metric_df = res_df[res_df['metric'] == metric]
        if max_checkpoint is None and checkpoint_tokens_map is None:
            plt.axvline(39063, color='white', linestyle='--')
            plt.axvline(39063*2, color='white', linestyle='--')
        sns.lineplot(data=metric_df, x=x_key, hue='model', y='score', marker='o', hue_order=sorted_models, palette=palette, ax=axes[idx])
        axes[idx].set_title(f'{task} - {metric_name if metric_name is not None else metric}')
    plt.savefig(f'{output_path}.png') 
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
    parser.add_argument('-t', '--downstream_task', choices=['sentiment', 'complexity', 'pos_tagging'])
    parser.add_argument('-m', '--model_name', choices=['bert', 'gpt'])
    parser.add_argument('-d', '--model_size')#, default='medium', choices=['small', 'medium', 'base'])
    parser.add_argument('-s', '--model_seed')
    parser.add_argument('-a', '--average_random', action='store_true')
    parser.add_argument('-n', '--num_tokens_map', action='store_true')
    args = parser.parse_args()

    if args.num_tokens_map:
        num_tokens_dir = 'data/num_training_tokens'
        checkpoint_tokens_map = load_checkpoint_tokens_map(num_tokens_dir)
    else:
        checkpoint_tokens_map = None

    if args.model_seed is not None:
        output_path = f'results/{args.model_name}/seed_{args.model_seed}/{args.model_name}_{args.downstream_task}'   
    else:
        output_path = f'results/{args.model_name}_{args.model_size}_{args.downstream_task}'
    if args.num_tokens_map:
        output_path = 'num_tokens_' + output_path 

    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)

    average_metrics = True if args.downstream_task == 'sentiment' else False
    models_dir = f'models/downstream_tasks/{args.model_name}/{args.downstream_task}'
    res_df = get_task_results(models_dir, args.downstream_task, args.model_size, model_seed=args.model_seed, average_random=args.average_random, average_metrics=average_metrics)
    plot_results(res_df, args.downstream_task, output_path, checkpoint_tokens_map, average_random=args.average_random, )#, metric_name='avg f1')

if __name__ == '__main__':
    main()