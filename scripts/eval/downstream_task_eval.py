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

def get_task_results(downstream_task, model_size, average_metrics=False):
    models_dir = f'models/downstream_tasks/{downstream_task}'
    res_df = []
    for model_name in os.listdir(models_dir):
        if model_size in model_name:
            model_df = get_model_results(models_dir, model_name, downstream_task, average_metrics)
            model_df['model'] = model_name
            res_df.append(model_df)
    res_df = pd.concat(res_df)
    return res_df

def plot_results(res_df, task, output_path, max_checkpoint=None, metric_name=None):
    sorted_models = sorted(list(res_df['model'].unique()), reverse=True)
    if max_checkpoint is not None:
        res_df = res_df[res_df['checkpoint'] <= max_checkpoint]
    for metric in res_df['metric'].unique():
        metric_df = res_df[res_df['metric'] == metric]
        plt.figure()
        if max_checkpoint is None:
            plt.axvline(39063, color='white', linestyle='--')
            plt.axvline(39063*2, color='white', linestyle='--')
        sns.lineplot(data=metric_df, x='checkpoint', hue='model', y='score', marker='o', hue_order=sorted_models, palette='Paired')
        plt.title(f'{task} - {metric_name if metric_name is not None else metric}')
        plt.savefig(f'{output_path}_{metric}.png') 
        plt.show() 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--downstream_task', choices=['sentiment', 'complexity', 'pos_tagging'])
    parser.add_argument('-s', '--model_size', choices=['small', 'medium', 'base'])
    parser.add_argument('-o', '--output_path', type=str)
    args = parser.parse_args()

    average_metrics = True if args.downstream_task == 'sentiment' else False
    res_df = get_task_results(args.downstream_task, args.model_size, average_metrics=average_metrics)
    plot_results(res_df, args.downstream_task, args.output_path, metric_name='avg f1')

if __name__ == '__main__':
    main()