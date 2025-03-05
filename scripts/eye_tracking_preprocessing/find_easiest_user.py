import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

sns.set_style('darkgrid')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--model_seed', type=int, choices=[42, 755, 995])
    args = parser.parse_args()

    eye_tracking_results_path = f'data/eye_tracking_data/user_performances_random_init_{args.model_seed}.csv'
    df = pd.read_csv(eye_tracking_results_path)
    last_epoch_df = df[df['epoch'] == 200][['user_id', 'fold', 'feature', 'score']]


    # Average scores on last epoch and across all epochs

    last_epoch_avg_score = last_epoch_df[['user_id', 'score']].groupby('user_id').mean()
    last_epoch_avg_score['epoch'] = 'last'
    all_epochs_avg_score = df[['user_id', 'score']].groupby('user_id').mean()
    all_epochs_avg_score['epoch'] = 'avg'
    avg_user_scores = pd.concat([last_epoch_avg_score, all_epochs_avg_score])

    # User with max average score at last epoch

    max_user =  (last_epoch_df[['user_id', 'score']].groupby('user_id').mean()).idxmax().item()
    print(f'Easiest user = {max_user}')

    sns.lineplot(avg_user_scores, x='user_id', y='score', hue='epoch');
    plt.axvline(max_user, color='dimgray', linestyle='--')
    plt.text(max_user-0.5, 
            avg_user_scores['score'].max()-0.1,  
            f'User {max_user}', 
            color='dimgray', 
            ha='right', 
            va='bottom', 
            fontsize=12)
    plt.savefig(f'data/eye_tracking_data/easiest_user_{args.model_seed}.png', bbox_inches='tight')


if __name__ == '__main__':
    main()