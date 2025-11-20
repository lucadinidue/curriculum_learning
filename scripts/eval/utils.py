import seaborn as sns
import json
import os

def get_last_checkpoint(model_dir):
    trainer_state_path = os.path.join(model_dir, 'trainer_state.json')
    with open(trainer_state_path, 'r') as src_file:
        trainer_state = json.load(src_file)
    return trainer_state['max_steps']


def get_seaborn_palette(num_colors):    
    if num_colors <= 10:
        return sns.color_palette("Paired", num_colors)
    
    paired_palette = sns.color_palette("Paired", 10)

    extra_colors = [
        (0.6, 0.85, 0.8), (0.3, 0.6, 0.55),     # Light and dark seafoam
        (1.0, 0.8, 0.7), (0.85, 0.45, 0.3),     # Light peach and deep terracotta
        (0.8, 0.75, 0.9), (0.6, 0.5, 0.75),     # Light and dark lavender
    ]

    extended_palette = paired_palette + extra_colors[:num_colors-10]
    return extended_palette

def map_checkpoints_to_tokens(res_df, checkpoint_tokens_map):
    def map_row(row):
        curriculum = '_'.join(row['model'].split('_')[5:])
        if 'rand' in curriculum and not 'random_' in curriculum:
            curriculum = curriculum.replace('rand', 'random_')
        if curriculum == '': # for the case with no seed specified
            curriculum = row['model']
        checkpoint = str(int(row['checkpoint']))
        num_tokens = None
        if curriculum in checkpoint_tokens_map:
            if checkpoint in checkpoint_tokens_map[curriculum]:
                num_tokens = checkpoint_tokens_map[curriculum][checkpoint]
                return num_tokens
        else:
            raise ValueError(f'Curriculum {curriculum} not found in checkpoint_tokens_map')
        return None
    
    res_df['num_training_tokens'] = res_df.apply(map_row, axis=1)
    res_df = res_df[res_df['num_training_tokens'].notnull()]
    return res_df

def normalize_model_name(model_name, model_seed, average_random):
    if model_seed is None:
        model_name = '_'.join(model_name.split('_')[5:])
    if average_random:
        if 'rand' in model_name or 'orig' in model_name:
            model_name = 'random'
    return model_name

def load_checkpoint_tokens_map(src_dir):
    checkpoint_tokens_map = {}
    for curriculum_file_name in os.listdir(src_dir):
        curriculum = curriculum_file_name.split('.')[0]
        file_path = os.path.join(src_dir, curriculum_file_name)
        with open(file_path, 'r') as src_file:
            curriculum_map = json.load(src_file)
        checkpoint_tokens_map[curriculum] = curriculum_map
    return checkpoint_tokens_map